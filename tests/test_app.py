import sys
from pathlib import Path

import pytest
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
