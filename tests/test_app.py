import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, ingestion, main as app_module, retrieval
from langchain_core.documents import Document


class FakeChain:
    def __init__(self, answer="mock answer"):
        self.answer = answer
        self.last_input = None

    def invoke(self, payload):
        self.last_input = payload["input"]
        return {"answer": self.answer}


class FakeLLMResponse:
    def __init__(self, content):
        self.content = content


@pytest.fixture(autouse=True)
def reset_app_state(monkeypatch):
    with app_module.STATE_LOCK:
        app_module.SESSION_STATES.clear()
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    yield
    with app_module.STATE_LOCK:
        app_module.SESSION_STATES.clear()


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_home_page_serves_ui(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "LinkMind AI" in response.text


def test_health_reports_runtime_config(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == app_module.APP_VERSION
    assert response.json()["openai_key_configured"] is True
    assert response.json()["retrieval"] == "local-keyword"


def test_ingest_rejects_invalid_url(client):
    response = client.post(
        "/api/ingest",
        json={"url": "not-a-url", "session_id": "session-a"},
    )

    assert response.status_code == 422


def test_ingest_requires_openai_key(client, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    response = client.post(
        "/api/ingest",
        json={"url": "https://example.com", "session_id": "session-a"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "OpenAI API Key not configured."


def test_ingest_stores_chain_per_session(client, monkeypatch):
    fake_chain = FakeChain()

    monkeypatch.setattr(
        app_module,
        "load_and_index_urls",
        lambda url: ({"vector": url}, "Success"),
    )
    monkeypatch.setattr(app_module, "setup_qa_chain", lambda vector_store: fake_chain)

    response = client.post(
        "/api/ingest",
        json={"url": "https://example.com/docs", "session_id": "session-a"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["url"] == "https://example.com/docs"
    assert app_module.SESSION_STATES["session-a"]["qa_chain"] is fake_chain
    assert app_module.SESSION_STATES["session-a"]["active_url"] == "https://example.com/docs"


def test_html_to_text_removes_noise_and_keeps_page_content():
    html = """
    <html>
      <head>
        <title>LinkMind AI</title>
        <meta name="description" content="Talk to the web.">
        <style>.hidden { color: red; }</style>
      </head>
      <body>
        <script>alert("nope")</script>
        <h1>OpenAI Integration</h1>
        <p>Powered by gpt-4o-mini and text-embedding-3-small.</p>
      </body>
    </html>
    """

    text = ingestion.html_to_text(html)

    assert "LinkMind AI" in text
    assert "Talk to the web." in text
    assert "gpt-4o-mini" in text
    assert "text-embedding-3-small" in text
    assert "alert" not in text
    assert ".hidden" not in text


def test_local_retrieval_selects_relevant_context(monkeypatch):
    docs = [
        Document(page_content="Installation steps for a FastAPI app.", metadata={}),
        Document(
            page_content="OpenAI Integration: Powered by gpt-4o-mini and text-embedding-3-small.",
            metadata={},
        ),
    ]

    monkeypatch.setattr(retrieval.ChatOpenAI, "invoke", lambda self, messages: None)
    qa = retrieval.LocalRetrievalQA(docs)

    result = qa.extractive_fallback("what is powered by this project", qa.retrieve("powered by"))

    assert "gpt-4o-mini" in result
    assert "text-embedding-3-small" in result


def test_local_retrieval_handles_hindi_context_and_hinglish_question(monkeypatch):
    docs = [
        Document(page_content="यह पेज LinkMind AI के बारे में सामान्य जानकारी देता है।", metadata={}),
        Document(page_content="इस ऐप को Vipin ने विकसित किया है।", metadata={}),
    ]

    monkeypatch.setattr(
        retrieval.ChatOpenAI,
        "invoke",
        lambda self, messages: FakeLLMResponse("किसने बनाया विकसित किया"),
    )
    qa = retrieval.LocalRetrievalQA(docs)

    result = qa.extractive_fallback("isko kisne banaya hai", qa.retrieve("isko kisne banaya hai"))

    assert "Vipin" in result


def test_query_expansion_falls_back_when_openai_is_unavailable(monkeypatch):
    def raise_connection_error(self, messages):
        raise retrieval.APIConnectionError(request=None)

    monkeypatch.setattr(retrieval.ChatOpenAI, "invoke", raise_connection_error)
    qa = retrieval.LocalRetrievalQA([
        Document(page_content="यह पेज हिंदी जानकारी रखता है।", metadata={}),
    ])

    assert qa.expand_query("hindi jankari") == "hindi jankari"


def test_ingest_uses_jina_fallback_for_forbidden_url(monkeypatch):
    def blocked_fetch(url):
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError(response=response)

    fallback_doc = Document(
        page_content="Fallback page title. Fallback description with enough readable content to index.",
        metadata={"source": "https://blocked.example"},
    )

    monkeypatch.setattr(ingestion, "fetch_url_document", blocked_fetch)
    monkeypatch.setattr(ingestion, "fetch_url_document_via_jina", lambda url: fallback_doc)

    docs, message = ingestion.load_and_index_urls("https://blocked.example")

    assert message == "Success"
    assert docs
    assert "Fallback page title" in docs[0].page_content


def test_ingest_uses_microlink_if_reader_fallback_fails(monkeypatch):
    def blocked_fetch(url):
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError(response=response)

    fallback_doc = Document(
        page_content="Microlink title. Microlink description with enough readable content to index.",
        metadata={"source": "https://blocked.example"},
    )

    monkeypatch.setattr(ingestion, "fetch_url_document", blocked_fetch)
    monkeypatch.setattr(ingestion, "fetch_url_document_via_jina", lambda url: (_ for _ in ()).throw(ValueError("reader failed")))
    monkeypatch.setattr(ingestion, "fetch_url_document_via_microlink", lambda url: fallback_doc)

    docs, message = ingestion.load_and_index_urls("https://blocked.example")

    assert message == "Success"
    assert docs
    assert "Microlink title" in docs[0].page_content


def test_chat_requires_ingested_url_first(client):
    response = client.post(
        "/api/chat",
        json={"query": "what is this?", "session_id": "missing-session"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please process a URL first."


def test_chat_uses_matching_session_chain(client):
    chain_a = FakeChain("answer from a")
    chain_b = FakeChain("answer from b")
    with app_module.STATE_LOCK:
        app_module.SESSION_STATES["session-a"] = {
            "vector_store": object(),
            "qa_chain": chain_a,
            "active_url": "https://a.example",
        }
        app_module.SESSION_STATES["session-b"] = {
            "vector_store": object(),
            "qa_chain": chain_b,
            "active_url": "https://b.example",
        }

    response = client.post(
        "/api/chat",
        json={"query": "  powered by?  ", "session_id": "session-b"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "answer from b"}
    assert chain_b.last_input == "powered by?"
    assert chain_a.last_input is None


def test_system_prompt_keeps_exact_details_and_vipin_identity():
    prompt = retrieval.SYSTEM_PROMPT

    assert "developed by Vipin" in prompt
    assert "exact model names" in prompt
    assert "Do not say 'I don't know'" in prompt


def test_ingest_and_chat_with_real_pipeline(client, monkeypatch):
    """Exercise module wiring while replacing only the HTTP and LLM calls."""
    page = requests.Response()
    page.status_code = 200
    page.headers["content-type"] = "text/html"
    page._content = (
        b"<html><body><script>discard this script</script>"
        b"<p>LinkMind AI was developed by Vipin using FastAPI and gpt-4o-mini.</p>"
        b"<p>" + b"The app answers questions about ingested webpages. " * 60
        + b"</p></body></html>"
    )
    http_calls = []
    llm_calls = []

    def fake_get(url, **kwargs):
        http_calls.append((url, kwargs))
        return page

    def fake_invoke(self, messages):
        llm_calls.append(messages)
        if len(llm_calls) == 1:
            return FakeLLMResponse("developed Vipin")
        assert "Vipin" in messages[0].content
        assert messages[1].content == "Who developed LinkMind AI?"
        return FakeLLMResponse("Vipin developed LinkMind AI.")

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    monkeypatch.setattr(retrieval.ChatOpenAI, "invoke", fake_invoke)

    response = client.post("/api/ingest", json={
        "url": "https://example.com/article", "session_id": "pipeline",
    })
    assert response.status_code == 200
    chunks = app_module.SESSION_STATES["pipeline"]["vector_store"]
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    assert all(chunk.metadata["source"] == "https://example.com/article" for chunk in chunks)
    assert "discard this script" not in " ".join(chunk.page_content for chunk in chunks)
    assert len(http_calls) == 1
    assert http_calls[0][0] == "https://example.com/article"
    assert http_calls[0][1]["timeout"] == 25

    response = client.post("/api/chat", json={
        "query": "Who developed LinkMind AI?", "session_id": "pipeline",
    })
    assert response.status_code == 200
    assert response.json() == {"answer": "Vipin developed LinkMind AI."}
    assert len(llm_calls) == 2


def test_chat_returns_extracted_text_when_openai_fails(client, monkeypatch):
    def unavailable(self, messages):
        raise retrieval.APIConnectionError(request=None)

    monkeypatch.setattr(retrieval.ChatOpenAI, "invoke", unavailable)
    chain = retrieval.LocalRetrievalQA([
        Document(page_content="Vipin developed LinkMind AI using FastAPI."),
    ])
    with app_module.STATE_LOCK:
        app_module.SESSION_STATES["offline"] = {"qa_chain": chain}

    response = client.post("/api/chat", json={
        "query": "Who developed LinkMind AI?", "session_id": "offline",
    })
    assert response.status_code == 200
    assert "Vipin developed LinkMind AI" in response.json()["answer"]


@pytest.mark.parametrize("status_code", [401, 403, 429])
def test_real_fetch_fallback_order(monkeypatch, status_code):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        response = requests.Response()
        if len(calls) == 1:
            response.status_code = status_code
        elif len(calls) == 2:
            response.status_code = 503
        else:
            response.status_code = 200
            response._content = (
                b'{"status":"success","data":{"title":"LinkMind AI",'
                b'"description":"A readable preview with enough text for indexing the webpage."}}'
            )
            assert kwargs["params"] == {"url": "https://blocked.example"}
        return response

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    docs, message = ingestion.load_and_index_urls("https://blocked.example")
    assert message == "Success"
    assert "readable preview" in docs[0].page_content
    assert calls == [
        "https://blocked.example",
        "https://r.jina.ai/https://blocked.example",
        "https://api.microlink.io/",
    ]


@pytest.mark.parametrize("failure", ["empty", "connection", "not_found"])
def test_failed_ingest_preserves_existing_session(client, monkeypatch, failure):
    existing = {"qa_chain": FakeChain("previous answer"), "active_url": "https://old.example"}
    with app_module.STATE_LOCK:
        app_module.SESSION_STATES["existing"] = existing

    def fake_get(url, **kwargs):
        if failure == "connection":
            raise requests.ConnectionError("Connection unavailable")
        response = requests.Response()
        response.status_code = 404 if failure == "not_found" else 200
        response._content = b""
        return response

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    response = client.post("/api/ingest", json={
        "url": "https://example.com/empty", "session_id": "existing",
    })
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Failed to ingest URL:")
    assert app_module.SESSION_STATES["existing"] is existing


@pytest.mark.parametrize("entrypoint", ["app:app", "src.main:app"])
def test_startup_and_assets_outside_project_directory(tmp_path, entrypoint):
    """Both Uvicorn imports must serve the UI even when cwd is elsewhere."""
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = "test-key"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(config.BASE_DIR), *[str(Path(path).resolve()) for path in sys.path if path]]
    )
    check = '''
import sys
from fastapi.testclient import TestClient
from uvicorn.importer import import_from_string
from src.main import app

assert import_from_string(sys.argv[1]) is app
with TestClient(app) as client:
    assert "LinkMind AI" in client.get("/").text
    for path in ("/", "/static/script.js", "/static/style.css", "/api/health"):
        assert client.get(path).status_code == 200, path
    assert client.get("/api/health").json()["openai_key_configured"] is True
'''
    result = subprocess.run(
        [sys.executable, "-c", check, entrypoint],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
