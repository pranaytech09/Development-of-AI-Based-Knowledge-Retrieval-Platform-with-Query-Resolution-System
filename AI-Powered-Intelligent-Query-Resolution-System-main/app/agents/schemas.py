"""Structured outputs used by agent nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """LLM structured output for query rewriting / clarity checks."""

    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable.")
    questions: list[str] = Field(description="List of rewritten, self-contained questions.")
    clarification_needed: str = Field(description="Explanation if the question is unclear.")
