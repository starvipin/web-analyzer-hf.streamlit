"""FastAPI entry point: connect HTTP requests to ingestion and retrieval."""

import logging
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .config import APP_VERSION, LLM_MODEL
from .ingestion import load_and_index_urls
from .retrieval import setup_qa_chain
from .schemas import ChatRequest, URLRequest

logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")

# Template directory setup
templates = Jinja2Templates(directory=config.BASE_DIR / "templates")

# In-memory session state. Good for local/demo use; move this to durable storage
# before running multiple worker processes or a public deployment.
SESSION_STATES = {}
STATE_LOCK = Lock()



@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_key_configured": bool(config.OPENAI_API_KEY),
        "llm_model": LLM_MODEL,
        "retrieval": "local-keyword",
    }


@app.post("/api/ingest")
async def ingest_url(req: URLRequest):
    if not config.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured.")

    url = str(req.url)

    chunks, msg = load_and_index_urls(url)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"Failed to ingest URL: {msg}")

    qa_chain = setup_qa_chain(chunks)

    with STATE_LOCK:
        SESSION_STATES[req.session_id] = {
            "vector_store": chunks,
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
