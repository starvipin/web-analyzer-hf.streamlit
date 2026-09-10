"""Validation for the JSON accepted by the ingest and chat endpoints."""

from pydantic import AnyHttpUrl, BaseModel, Field

class URLRequest(BaseModel):
    url: AnyHttpUrl
    session_id: str = Field(default="default", min_length=1, max_length=128)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
