"""Unit tests for DocumentRetriever-backed agent tools."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.agents.tools import ToolFactory
from app.core.config import Settings
from app.schemas.rag import SearchResult


def test_search_documents_formats_hits() -> None:
    retriever = MagicMock()
    retriever.search.return_value = [
        SearchResult(
            document_id="doc_1",
            score=0.9,
            document="Leave policy allows 12 weeks.",
            metadata={"filename": "hr.pdf"},
        )
    ]
    settings = Settings(retrieval_score_threshold=0.3, default_retrieval_k=3, chunk_separator="\n---\n")
    tools = ToolFactory(retriever=retriever, settings=settings).create_tools()
    assert len(tools) == 1
    assert tools[0].name == "search_documents"

    output = tools[0].invoke({"query": "parental leave", "limit": 3})
    retriever.search.assert_called_once_with("parental leave", top_k=3)
    assert "hr.pdf" in output
    assert "Leave policy" in output
    assert "Score: 0.9000" in output


def test_search_documents_filters_low_scores() -> None:
    retriever = MagicMock()
    retriever.search.return_value = [
        SearchResult(document_id="a", score=0.1, document="noise", metadata={"filename": "a.pdf"})
    ]
    settings = Settings(retrieval_score_threshold=0.5)
    tool = ToolFactory(retriever=retriever, settings=settings).create_tools()[0]
    assert tool.invoke({"query": "x"}) == "NO_RELEVANT_CHUNKS"
