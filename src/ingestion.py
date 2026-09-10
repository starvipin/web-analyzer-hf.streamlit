"""Fetch a URL, clean page text, and split it into searchable chunks."""

import logging

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .config import JINA_READER_URL, MICROLINK_API_URL, REQUEST_HEADERS

logger = logging.getLogger(__name__)

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


def fetch_url_document_via_jina(url: str) -> Document:
    response = requests.get(f"{JINA_READER_URL}{url}", timeout=40)
    response.raise_for_status()
    text = response.text.strip()

    if len(text) <= 50:
        raise ValueError("Jina Reader fallback did not return enough readable content.")

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


def load_and_index_urls(url: str):
    if not config.OPENAI_API_KEY:
        return None, "OpenAI API Key not configured."

    try:
        docs = [fetch_url_document(url)]
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"Ingestion HTTP error for {url}: {e}", exc_info=True)
        if status_code in (401, 403, 429):
            try:
                docs = [fetch_url_document_via_jina(url)]
            except Exception as jina_error:
                logger.error(f"Jina Reader fallback failed for {url}: {jina_error}", exc_info=True)
                try:
                    docs = [fetch_url_document_via_microlink(url)]
                except Exception as fallback_error:
                    logger.error(f"Microlink fallback failed for {url}: {fallback_error}", exc_info=True)
                    return None, (
                        f"URL returned HTTP {status_code}. The site is blocking automated access, "
                        "and both reader and preview fallback extraction failed."
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
