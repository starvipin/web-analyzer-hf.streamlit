import os
import logging
from threading import Lock
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

# --- LangChain Imports ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini" 
EMBEDDING_MODEL = "text-embedding-3-small"

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
def load_and_index_urls(url: str):
    if not openai_api_key:
        return None, "OpenAI API Key not configured."

    try:
        loader = WebBaseLoader([url], requests_per_second=2, continue_on_failure=True)
        loader.requests_kwargs = {'timeout': 20}
        docs = loader.load()

        docs = [doc for doc in docs if len(doc.page_content) > 50]
        if not docs:
            return None, "Content is too short or empty."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, "Success"
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return None, str(e)

def setup_qa_chain(vector_store):
    system_prompt = (
    "You are LinkMind AI, developed by Vipin.\n"
    "Answer questions about the user's ingested link using ONLY the provided context.\n"
    "Understand casual, misspelled, or short questions by mapping them to the closest meaning in context.\n"
    "Always use exact model names, product names, versions, and technical details — never summarize them.\n"
    "If the answer isn't in the context, briefly state what topics you can help with from this link "
    "and ask for a more specific question. Do not say 'I don't know'.\n"
    "If asked about this app or its developer, say it was developed by Vipin.\n\n"
    "CONTEXT:\n{context}"
)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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
