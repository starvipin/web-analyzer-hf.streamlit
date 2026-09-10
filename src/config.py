"""Shared runtime settings and paths relative to the project root."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["USER_AGENT"] = "MyWebAnalyzerApp/1.0"
logging.basicConfig(level=logging.INFO)

LLM_MODEL = "gpt-4o-mini"
APP_VERSION = "2026-05-28-1"
OPENAI_TIMEOUT_SECONDS = 45
OPENAI_MAX_RETRIES = 3
REQUEST_HEADERS = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
MICROLINK_API_URL = "https://api.microlink.io/"
JINA_READER_URL = "https://r.jina.ai/"
