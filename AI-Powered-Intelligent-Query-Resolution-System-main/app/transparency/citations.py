"""Extract citation filenames and source chunk payloads from agent contexts."""

from __future__ import annotations

import re
from dataclasses import dataclass


_FILE_RE = re.compile(r"File Name:\s*(.+)", re.IGNORECASE)
_DOC_ID_RE = re.compile(r"Document ID:\s*(.+)", re.IGNORECASE)
_SCORE_RE = re.compile(r"Score:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_CONTENT_RE = re.compile(r"Content:\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SourceChunk:
    """A retrieved passage used to ground an answer."""

    document_id: str
    filename: str
    score: float | None
    content: str


def extract_citations(contexts: list[str]) -> list[str]:
    """Return unique source filenames mentioned in retrieval contexts."""
    names: list[str] = []
    for context in contexts:
        match = _FILE_RE.search(context)
        if match:
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def extract_source_chunks(contexts: list[str]) -> list[SourceChunk]:
    """Parse tool-formatted contexts into structured source chunks."""
    chunks: list[SourceChunk] = []
    for context in contexts:
        doc_id_match = _DOC_ID_RE.search(context)
        file_match = _FILE_RE.search(context)
        score_match = _SCORE_RE.search(context)
        content_match = _CONTENT_RE.search(context)
        chunks.append(
            SourceChunk(
                document_id=(doc_id_match.group(1).strip() if doc_id_match else ""),
                filename=(file_match.group(1).strip() if file_match else "unknown"),
                score=(float(score_match.group(1)) if score_match else None),
                content=(content_match.group(1).strip() if content_match else context.strip()),
            )
        )
    return chunks
