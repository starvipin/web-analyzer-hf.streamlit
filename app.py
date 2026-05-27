import os
import logging
from threading import Lock
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
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
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini" 
EMBEDDING_MODEL = "text-embedding-3-small"
REQUEST_HEADERS = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}

SYSTEM_PROMPT = (
    "You are LinkMind AI, developed by Vipin.\n"
    "Answer questions about the user's ingested link using ONLY the provided context.\n"
    "Understand casual, misspelled, or short questions by mapping them to the closest meaning in context.\n"
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


def load_and_index_urls(url: str):
    if not openai_api_key:
        return None, "OpenAI API Key not configured."

    try:
        docs = [fetch_url_document(url)]

        docs = [doc for doc in docs if len(doc.page_content) > 50]
        if not docs:
            return None, "Content is too short or empty."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, "Success"
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"Ingestion HTTP error for {url}: {e}", exc_info=True)
        return None, f"URL returned HTTP {status_code}. The site may be blocking automated access."
    except requests.RequestException as e:
        logger.error(f"Ingestion connection error for {url}: {e}", exc_info=True)
        return None, f"Could not connect to the URL from the server: {e}"
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return None, str(e)

def setup_qa_chain(vector_store):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, openai_api_key=openai_api_key, max_tokens=300)
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    # FIXED: Modern FastAPI/Starlette syntax to prevent "unhashable type: 'dict'"
    return templates.TemplateResponse(request=request, name="index.html")

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
