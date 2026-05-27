import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Load .env variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- LangChain Imports ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- ENV FIXES ---
os.environ["USER_AGENT"] = "MyWebAnalyzerApp/1.0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini" 
EMBEDDING_MODEL = "text-embedding-3-small"

app = FastAPI()

# Template directory setup
templates = Jinja2Templates(directory="templates")

# Global variables to act as simple session state
GLOBAL_STATE = {
    "vector_store": None,
    "qa_chain": None,
    "active_url": None
}

# --- Request Models ---
class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    query: str

# --- Helper Functions ---
def load_and_index_urls(url: str):
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
        "You are an assistant designed to answer questions based ONLY on the provided context.\n"
        "Carefully review the following context.\n"
        "If the answer to the question is present in the context, provide that answer cleanly.\n"
        "If the answer is not found in the context, you MUST respond with 'I don't know'.\n"
        "Do not add any information that is not explicitly stated in the context.\n\n"
        "CONTEXT:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, openai_api_key=openai_api_key, max_tokens=300)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
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
    
    vectorstore, msg = load_and_index_urls(req.url)
    if not vectorstore:
        raise HTTPException(status_code=400, detail=f"Failed to ingest URL: {msg}")
    
    qa_chain = setup_qa_chain(vectorstore)
    
    GLOBAL_STATE["vector_store"] = vectorstore
    GLOBAL_STATE["qa_chain"] = qa_chain
    GLOBAL_STATE["active_url"] = req.url
    
    return {"status": "success", "message": "System Ready!", "url": req.url}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GLOBAL_STATE["qa_chain"]:
        raise HTTPException(status_code=400, detail="Please process a URL first.")
    
    try:
        result = GLOBAL_STATE["qa_chain"].invoke({"input": req.query})
        answer = result.get("answer", "Sorry, I couldn't find the answer.")
        
        # Extracting brief source context if needed
        sources = [doc.page_content[:150] + "..." for doc in result.get("context", [])]
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))