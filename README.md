---
title: LinkMind AI
emoji: 🔗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# LinkMind AI

LinkMind AI is a FastAPI-based web application developed by Vipin. It lets a user paste a public URL, extracts readable page content, and then answers questions about that specific link using the ingested context.

Live Space: https://sainivipin-linkmind-ai.hf.space

## What It Does

- Accepts a website/article URL from the browser UI.
- Fetches readable text from the page.
- Splits the page content into smaller chunks.
- Retrieves the most relevant chunks for each user question.
- Uses OpenAI `gpt-4o-mini` to answer only from the retrieved link context.
- Supports Hindi, English, and Hinglish questions.
- Keeps each browser session isolated with a client-side `session_id`.
- Falls back to local extractive answers if OpenAI chat is temporarily unavailable.

## Tech Stack

- Backend: FastAPI
- Frontend: HTML, CSS, JavaScript
- Templates: Jinja2
- Web extraction: `requests`, BeautifulSoup
- Text chunking: LangChain `RecursiveCharacterTextSplitter`
- LLM: OpenAI `gpt-4o-mini`
- Deployment: Docker on Hugging Face Spaces
- CI/CD: GitHub Actions
- Testing: pytest, FastAPI TestClient
- Package manager: uv

## How The App Works

1. The user enters a URL in the frontend.
2. The frontend validates the URL and sends it to `POST /api/ingest`.
3. The backend tries to fetch the page directly using browser-like request headers.
4. If the target site blocks server access with `401`, `403`, or `429`, the backend tries fallbacks:
   - Jina Reader via `https://r.jina.ai/`
   - Microlink metadata extraction
5. The extracted content is cleaned with BeautifulSoup.
6. The content is split into chunks with `RecursiveCharacterTextSplitter`.
7. The chunks are stored in memory for that browser session.
8. The user asks a question through `POST /api/chat`.
9. The app expands the query for multilingual retrieval when OpenAI is available.
10. Local keyword and character n-gram retrieval selects relevant chunks.
11. `gpt-4o-mini` answers from the selected context.
12. If the OpenAI chat call fails, the app returns a local extractive fallback answer from the retrieved text.

## API Endpoints

### `GET /`

Serves the main LinkMind AI web interface.

### `GET /api/health`

Returns runtime health and deployment information.

Example:

```json
{
  "status": "ok",
  "version": "2026-05-28-1",
  "openai_key_configured": true,
  "llm_model": "gpt-4o-mini",
  "retrieval": "local-keyword"
}
```

### `POST /api/ingest`

Processes a URL and prepares it for Q&A.

Request:

```json
{
  "url": "https://example.com/article",
  "session_id": "browser-session-id"
}
```

### `POST /api/chat`

Answers a question about the active URL for the session. 

Request:

```json
{
  "query": "Is page ke bare me batao",
  "session_id": "browser-session-id"
}
```

Response:

```json
{
  "answer": "..."
}
```

## Retrieval Design

The project originally used OpenAI embeddings and FAISS, but Hugging Face Spaces had connection issues while creating embeddings. To make the deployed app reliable, ingestion was redesigned to avoid OpenAI embeddings.

Current retrieval:

- Unicode-aware token matching for English and Hindi text.
- Character n-gram matching for partial and mixed-language queries.
- Optional OpenAI query expansion for Hinglish, Hindi, and English questions.
- Local extractive fallback if OpenAI chat is unavailable.

This makes URL ingestion faster, cheaper, and more reliable on Hugging Face Spaces.

## Error Handling

Common cases handled:

- Missing OpenAI key
- Invalid URL
- Empty or too-short page content
- Site blocks direct fetch with `403`
- Jina Reader fallback fails
- Microlink fallback fails
- OpenAI chat connection failure
- Separate browser sessions asking questions about different links

If a website blocks all extraction methods, the app returns a clear error. A future improvement would be adding a manual paste-text option.

## Local Setup

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
```

Install dependencies:

```bash
uv sync --dev
```

Run locally:

```bash
uv run uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Run Tests

```bash
uv run pytest -q
```

The test suite covers:

- UI route
- Health endpoint
- URL validation
- Missing API key handling
- URL ingestion session storage
- HTML cleanup
- Hindi/Hinglish retrieval behavior
- 403 fallback flow
- Chat session isolation
- System prompt rules

## Docker

The Dockerfile runs the FastAPI app on port `7860`, which is required by Hugging Face Docker Spaces.

Build locally:

```bash
docker build -t linkmind-ai .
```

Run locally:

```bash
docker run -p 7860:7860 --env OPENAI_API_KEY=your_openai_api_key linkmind-ai
```

## Deployment

This project is deployed as a Hugging Face Docker Space:

```text
https://huggingface.co/spaces/sainivipin/LinkMind-AI
```

Required secrets:

- GitHub repository secret `HF_TOKEN`
- Hugging Face Space secret `OPENAI_API_KEY`

The GitHub Actions pipeline:

1. Runs tests.
2. Builds the Docker image.
3. Pushes the repo to the Hugging Face Space if tests and Docker build pass.

## Project Structure

```text
.
├── app.py                  # Compatibility entry point for HF / uvicorn
├── src/                    # Main Python application code
│   ├── __init__.py
│   ├── main.py             # FastAPI routes and session state
│   ├── config.py           # Environment, paths, and runtime settings
│   ├── schemas.py          # URL/chat request validation
│   ├── ingestion.py        # Fetch URL, clean HTML, and create chunks
│   └── retrieval.py        # Search chunks and generate answers
├── Dockerfile
├── README.md
├── pyproject.toml
├── uv.lock
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── tests/
│   └── test_app.py
├── docs/
│   └── CODE_WALKTHROUGH.md  # Interview explanation in Hinglish
└── .github/
    └── workflows/
        └── ci-cd.yml
```

Start reading at `src/main.py`, then follow `load_and_index_urls()` into
`src/ingestion.py` and `LocalRetrievalQA` into `src/retrieval.py`.
See the [interview code walkthrough](docs/CODE_WALKTHROUGH.md) for the request
flow and a file-by-file explanation.

The existing `uvicorn app:app` command still works through the small root
entry point. You can also run `uv run uvicorn src.main:app --reload`.
Docker, the Hugging Face port, dependency versions, and API URLs are unchanged.
Static files, templates, and `.env` are located relative to the project root.

## Limitations

- Some websites block all automated access.
- Microlink fallback may only provide title/description, not full article text.
- Session state is in memory, so it resets when the server restarts.
- Multi-worker production deployments would need Redis or a database for session storage.
- The current retrieval is local keyword/n-gram based, not vector embedding based.

## Future Improvements

- Add a manual paste-text mode for blocked pages.
- Add persistent session storage with Redis.
- Add source snippets in an optional developer/debug mode.
- Add file upload support for PDF or text documents.
- Add background cleanup for old sessions.
- Add better frontend status messages for fallback extraction stages.
