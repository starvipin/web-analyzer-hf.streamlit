import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


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
    monkeypatch.setattr(app_module, "openai_api_key", "test-key")
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
    monkeypatch.setattr(app_module, "openai_api_key", None)

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

    text = app_module.html_to_text(html)

    assert "LinkMind AI" in text
    assert "Talk to the web." in text
    assert "gpt-4o-mini" in text
    assert "text-embedding-3-small" in text
    assert "alert" not in text
    assert ".hidden" not in text


def test_local_retrieval_selects_relevant_context(monkeypatch):
    docs = [
        app_module.Document(page_content="Installation steps for a FastAPI app.", metadata={}),
        app_module.Document(
            page_content="OpenAI Integration: Powered by gpt-4o-mini and text-embedding-3-small.",
            metadata={},
        ),
    ]

    monkeypatch.setattr(app_module.ChatOpenAI, "invoke", lambda self, messages: None)
    qa = app_module.LocalRetrievalQA(docs)

    result = qa.extractive_fallback("what is powered by this project", qa.retrieve("powered by"))

    assert "gpt-4o-mini" in result
    assert "text-embedding-3-small" in result


def test_local_retrieval_handles_hindi_context_and_hinglish_question(monkeypatch):
    docs = [
        app_module.Document(page_content="यह पेज LinkMind AI के बारे में सामान्य जानकारी देता है।", metadata={}),
        app_module.Document(page_content="इस ऐप को Vipin ने विकसित किया है।", metadata={}),
    ]

    monkeypatch.setattr(
        app_module.ChatOpenAI,
        "invoke",
        lambda self, messages: FakeLLMResponse("किसने बनाया विकसित किया"),
    )
    qa = app_module.LocalRetrievalQA(docs)

    result = qa.extractive_fallback("isko kisne banaya hai", qa.retrieve("isko kisne banaya hai"))

    assert "Vipin" in result


def test_query_expansion_falls_back_when_openai_is_unavailable(monkeypatch):
    def raise_connection_error(self, messages):
        raise app_module.APIConnectionError(request=None)

    monkeypatch.setattr(app_module.ChatOpenAI, "invoke", raise_connection_error)
    qa = app_module.LocalRetrievalQA([
        app_module.Document(page_content="यह पेज हिंदी जानकारी रखता है।", metadata={}),
    ])

    assert qa.expand_query("hindi jankari") == "hindi jankari"


def test_ingest_uses_microlink_fallback_for_forbidden_url(monkeypatch):
    def blocked_fetch(url):
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError(response=response)

    fallback_doc = app_module.Document(
        page_content="Fallback page title. Fallback description with enough readable content to index.",
        metadata={"source": "https://blocked.example"},
    )

    monkeypatch.setattr(app_module, "fetch_url_document", blocked_fetch)
    monkeypatch.setattr(app_module, "fetch_url_document_via_microlink", lambda url: fallback_doc)

    docs, message = app_module.load_and_index_urls("https://blocked.example")

    assert message == "Success"
    assert docs
    assert "Fallback page title" in docs[0].page_content


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
    prompt = app_module.SYSTEM_PROMPT

    assert "developed by Vipin" in prompt
    assert "exact model names" in prompt
    assert "Do not say 'I don't know'" in prompt
