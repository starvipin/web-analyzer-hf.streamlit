import os
import logging
import re
from threading import Lock
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import AnyHttpUrl, BaseModel, Field

# --- Load .env variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- ENV FIXES ---
os.environ["USER_AGENT"] = "MyWebAnalyzerApp/1.0"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini" 
APP_VERSION = "2026-05-27-3"
OPENAI_TIMEOUT_SECONDS = 45
OPENAI_MAX_RETRIES = 3
REQUEST_HEADERS = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
MICROLINK_API_URL = "https://api.microlink.io/"

SYSTEM_PROMPT = (
    "You are LinkMind AI, developed by Vipin.\n"
    "Answer questions about the user's ingested link using ONLY the provided context.\n"
    "Understand casual, misspelled, or short questions by mapping them to the closest meaning in context.\n"
    "The context or user question may be in Hindi, English, or Hinglish. Answer in the same language style the user uses.\n"
    "If the page is in Hindi and the user asks in Hinglish, translate the meaning internally and answer from the Hindi context.\n"
    "Always use exact model names, product names, versions, and technical details - never summarize them.\n"
    "If the answer isn't in the context, briefly state what topics you can help with from this link "
    "and ask for a more specific question. Do not say 'I don't know'.\n"
    "If asked about this app or its developer, say it was developed by Vipin.\n\n"
    "CONTEXT:\n{context}"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template directory setup
templates = Jinja2Templates(directory="templates")

# In-memory session state. Good for local/demo use; move this to durable storage
# before running multiple worker processes or a public deployment.
SESSION_STATES = {}
STATE_LOCK = Lock()

# --- Request Models ---
class URLRequest(BaseModel):
    url: AnyHttpUrl
    session_id: str = Field(default="default", min_length=1, max_length=128)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=128)

# --- Helper Functions ---
def tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[\w\u0900-\u097F][\w\u0900-\u097F_-]+", text.lower(), flags=re.UNICODE)
        if len(word) > 1
    }


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return {normalized[index:index + n] for index in range(max(len(normalized) - n + 1, 0))}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    description_tag = soup.find("meta", attrs={"name": "description"})
    description = description_tag.get("content", "").strip() if description_tag else ""
    body_text = soup.get_text("\n", strip=True)

    return "\n\n".join(part for part in [title, description, body_text] if part)


def fetch_url_document(url: str) -> Document:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=25, allow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or "xml" in content_type:
        text = html_to_text(response.text)
    else:
        text = response.text

    return Document(page_content=text, metadata={"source": url})


def fetch_url_document_via_microlink(url: str) -> Document:
    response = requests.get(MICROLINK_API_URL, params={"url": url}, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise ValueError(payload.get("message") or "Microlink could not fetch this URL.")

    data = payload.get("data", {})
    parts = [
        data.get("title", ""),
        data.get("description", ""),
        data.get("publisher", ""),
        data.get("author", ""),
    ]
    text = "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())

    if len(text) <= 50:
        raise ValueError("Microlink fallback did not return enough readable content.")

    return Document(page_content=text, metadata={"source": url})


class LocalRetrievalQA:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_tokens = [tokenize(doc.page_content) for doc in docs]
        self.doc_ngrams = [char_ngrams(doc.page_content) for doc in docs]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            openai_api_key=openai_api_key,
            max_tokens=300,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )

    def expand_query(self, query: str) -> str:
        try:
            messages = [
                (
                    "system",
                    "Rewrite the user's search query into concise Hindi/English keywords for retrieval. "
                    "Include direct translations when the query is Hinglish. Return only keywords.",
                ),
                ("human", query),
            ]
            response = self.llm.invoke(messages)
            expanded = response.content.strip()
            return f"{query}\n{expanded}" if expanded else query
        except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError, AttributeError):
            logger.info("Query expansion unavailable; using original query.", exc_info=True)
            return query

    def retrieve(self, query: str, k: int = 8) -> list[Document]:
        expanded_query = self.expand_query(query)
        query_tokens = tokenize(expanded_query)
        query_ngrams = char_ngrams(expanded_query)
        if not query_tokens and not query_ngrams:
            return self.docs[:k]

        scored_docs = []
        query_lower = expanded_query.lower()
        for index, doc in enumerate(self.docs):
            token_score = len(query_tokens.intersection(self.doc_tokens[index]))
            ngram_score = len(query_ngrams.intersection(self.doc_ngrams[index])) / 8
            phrase_score = doc.page_content.lower().count(query_lower) * 3
            scored_docs.append((token_score + ngram_score + phrase_score, index, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        selected = [doc for score, _, doc in scored_docs if score > 0][:k]
        return selected or self.docs[:k]

    def extractive_fallback(self, query: str, context_docs: list[Document]) -> str:
        query_tokens = tokenize(query)
        sentences = []
        for doc in context_docs:
            sentences.extend(re.split(r"(?<=[.!?])\s+|\n+", doc.page_content))

        ranked = []
        for sentence in sentences:
            clean_sentence = " ".join(sentence.split())
            if len(clean_sentence) < 20:
                continue
            score = len(query_tokens.intersection(tokenize(clean_sentence)))
            ranked.append((score, clean_sentence))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best = [sentence for score, sentence in ranked if score > 0][:3]

        if best:
            return " ".join(best)
        if ranked:
            return " ".join(sentence for _, sentence in ranked[:2])
        return (
            "I can help with the information available in this link, such as its features, "
            "tech stack, models used, or setup details. Please ask a specific question."
        )

    def invoke(self, payload: dict) -> dict:
        query = payload["input"]
        context_docs = self.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in context_docs)

        try:
            messages = self.prompt.format_messages(context=context, input=query)
            response = self.llm.invoke(messages)
            return {"answer": response.content, "context": context_docs}
        except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError):
            logger.error("OpenAI chat failed; using local extractive fallback.", exc_info=True)
            return {"answer": self.extractive_fallback(query, context_docs), "context": context_docs}


def load_and_index_urls(url: str):
    if not openai_api_key:
        return None, "OpenAI API Key not configured."

    try:
        docs = [fetch_url_document(url)]
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"Ingestion HTTP error for {url}: {e}", exc_info=True)
        if status_code in (401, 403, 429):
            try:
                docs = [fetch_url_document_via_microlink(url)]
            except Exception as fallback_error:
                logger.error(f"Microlink fallback failed for {url}: {fallback_error}", exc_info=True)
                return None, (
                    f"URL returned HTTP {status_code}. The site is blocking automated access, "
                    "and fallback preview extraction also failed."
                )
        else:
            return None, f"URL returned HTTP {status_code}. The site may be blocking automated access."
    except requests.RequestException as e:
        logger.error(f"Ingestion connection error for {url}: {e}", exc_info=True)
        return None, f"Could not connect to the URL from the server: {e}"

    docs = [doc for doc in docs if len(doc.page_content) > 50]
    if not docs:
        return None, "Content is too short or empty."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits, "Success"

def setup_qa_chain(docs: list[Document]):
    return LocalRetrievalQA(docs)


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    # FIXED: Modern FastAPI/Starlette syntax to prevent "unhashable type: 'dict'"
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_key_configured": bool(openai_api_key),
        "llm_model": LLM_MODEL,
        "retrieval": "local-keyword",
    }

@app.post("/api/ingest")
async def ingest_url(req: URLRequest):
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured.")

    url = str(req.url)
    
    vectorstore, msg = load_and_index_urls(url)
    if not vectorstore:
        raise HTTPException(status_code=400, detail=f"Failed to ingest URL: {msg}")
    
    qa_chain = setup_qa_chain(vectorstore)
    
    with STATE_LOCK:
        SESSION_STATES[req.session_id] = {
            "vector_store": vectorstore,
            "qa_chain": qa_chain,
            "active_url": url
        }
    
    return {"status": "success", "message": "System Ready!", "url": url}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    with STATE_LOCK:
        state = SESSION_STATES.get(req.session_id)

    if not state or not state["qa_chain"]:
        raise HTTPException(status_code=400, detail="Please process a URL first.")
    
    try:
        result = state["qa_chain"].invoke({"input": req.query.strip()})
        answer = result.get("answer", "Sorry, I couldn't find the answer.")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
