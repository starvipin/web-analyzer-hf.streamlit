"""Keep the existing `uvicorn app:app` deployment command working."""

from src.main import app

__all__ = ["app"]
