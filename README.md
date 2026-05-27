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

LinkMind AI is a FastAPI web app developed by Vipin. Paste a URL, let the app ingest the page, and ask questions about the page content.

## Features

- URL ingestion with `WebBaseLoader`
- Chunking with `RecursiveCharacterTextSplitter`
- FAISS vector retrieval
- OpenAI chat responses with `gpt-4o-mini`
- OpenAI embeddings with `text-embedding-3-small`
- Browser session IDs so each user keeps their own active link context
- Docker deployment for Hugging Face Spaces

## Local Run

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
```

Install and run:

```bash
uv sync --dev
uv run uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Tests

```bash
uv run pytest -q
```

The tests mock URL ingestion and AI calls, so they verify routing, validation, session context, and prompt rules without spending API credits.

## Hugging Face Deployment

This repo is configured for Docker Spaces at:

https://huggingface.co/spaces/sainivipin/LinkMind-AI

Required secrets:

- GitHub repository secret `HF_TOKEN`: Hugging Face token used by GitHub Actions to sync this repo to the Space.
- Hugging Face Space secret `OPENAI_API_KEY`: OpenAI key used by the running app.

The GitHub Actions workflow runs tests, builds the Docker image, and deploys to Hugging Face only after CI passes on `main` or `master`.
