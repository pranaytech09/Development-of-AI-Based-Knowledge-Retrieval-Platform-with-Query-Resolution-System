"""Tests for transparency helpers."""

from __future__ import annotations

from app.transparency.citations import extract_citations, extract_source_chunks
from app.transparency.confidence import compute_retrieval_confidence


SAMPLE = (
    "Document ID: leave_0_abc\n"
    "File Name: hr_policy.pdf\n"
    "Score: 0.8200\n"
    "Content: Parental leave is 12 weeks."
)


def test_compute_retrieval_confidence() -> None:
    assert compute_retrieval_confidence([]) == 0.0
    assert compute_retrieval_confidence([SAMPLE]) == 0.82


def test_extract_citations_and_chunks() -> None:
    assert extract_citations([SAMPLE]) == ["hr_policy.pdf"]
    chunks = extract_source_chunks([SAMPLE])
    assert len(chunks) == 1
    assert chunks[0].filename == "hr_policy.pdf"
    assert chunks[0].document_id == "leave_0_abc"
    assert "12 weeks" in chunks[0].content
