"""API request/response schemas for query resolution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Ask a natural-language question against the knowledge base."""

    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        description="Optional LangGraph thread id for multi-turn continuity.",
    )


class SourceChunkPublic(BaseModel):
    """A passage used to ground the answer."""

    document_id: str
    filename: str
    score: float | None = None
    content: str


class QueryResponse(BaseModel):
    """Grounded answer with mandatory transparency fields."""

    answer: str
    thread_id: str
    clarification_needed: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    source_chunks: list[SourceChunkPublic] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Result of document ingestion."""

    filenames: list[str]
    chunks_indexed: int
    message: str
