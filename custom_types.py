"""
Pydantic models used across the RAG backend.
Clean, minimal set — no leftover Inngest scaffolding.
"""
from __future__ import annotations

import pydantic


class JobStatus(pydantic.BaseModel):
    """Tracks an in-progress or completed PDF ingestion job."""
    job_id: str
    status: str  # "processing" | "done" | "error" | "duplicate"
    source: str = ""
    total_pages: int = 0
    current_page: int = 0
    chunks_indexed: int = 0
    error: str = ""
    message: str = ""


class QueryPayload(pydantic.BaseModel):
    """Incoming query request from the frontend."""
    question: str
    top_k: int = 5
    history: list[dict] = pydantic.Field(default_factory=list)  # [{"role": "user"/"assistant", "content": str}]


class SearchResult(pydantic.BaseModel):
    """De-duplicated search result — one entry per unique passage."""
    contexts: list[str]
    sources: list[list[str]]  # sources[i] = list of source filenames that contain contexts[i]


class QueryResult(pydantic.BaseModel):
    """Final answer returned to the frontend."""
    answer: str
    sources: list[str]
    num_contexts: int
    cached: bool = False


class KnowledgeBaseInfo(pydantic.BaseModel):
    """Summary of what's currently indexed."""
    sources: list[str]
    total_chunks: int
