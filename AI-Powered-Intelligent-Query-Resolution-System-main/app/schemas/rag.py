"""Pydantic schemas for the document RAG pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single indexed chunk from an uploaded file."""

    filename: str
    file_type: str
    chunk_index: int
    content: str
    source_path: str = ""

    @property
    def document_id(self) -> str:
        """Stable identifier for vector-store indexing."""
        slug = re.sub(r"[^a-z0-9]+", "_", Path(self.filename).stem.lower()).strip("_")[:40]
        digest = hashlib.sha256(
            f"{self.filename}:{self.chunk_index}:{self.content}".encode()
        ).hexdigest()[:12]
        return f"{slug}_{self.chunk_index}_{digest}"

    def to_document_text(self) -> str:
        """Text passed to the embedding model."""
        return self.content

    def to_metadata(self) -> dict[str, str]:
        """Chroma-compatible flat metadata for filtering."""
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_index": str(self.chunk_index),
            "source_path": self.source_path,
        }


class SearchResult(BaseModel):
    """Single retrieval hit from the vector store."""

    document_id: str
    score: float
    document: str
    metadata: dict[str, str] = Field(default_factory=dict)
