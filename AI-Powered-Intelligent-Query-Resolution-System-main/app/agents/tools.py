"""LangChain tools backed by the existing DocumentRetriever (app.rag)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from app.agents.execution_logger import log_error, log_tool_end, log_tool_start
from app.core.config import Settings, get_settings
from app.rag.retriever import DocumentRetriever
from app.schemas.rag import SearchResult


class ToolFactory:
    """Builds retrieval tools that call :class:`DocumentRetriever`."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever or DocumentRetriever()
        self._settings = settings or get_settings()

    def _search_documents(self, query: str, limit: int | None = None) -> str:
        """Search indexed document chunks for evidence related to the user question.

        Use this as the primary retrieval step. Results include document IDs,
        file names, retrieval scores, and chunk text.

        Args:
            query: Focused search query with concrete keywords from the question.
            limit: Maximum number of chunks to return.
        """
        top_k = limit if limit is not None else self._settings.default_retrieval_k
        log_tool_start("search_documents", {"query": query, "limit": top_k})
        try:
            results = self._retriever.search(query, top_k=top_k)
            filtered = [
                hit
                for hit in results
                if hit.score >= self._settings.retrieval_score_threshold
            ]
            if not filtered:
                output = "NO_RELEVANT_CHUNKS"
                log_tool_end("search_documents", output)
                return output

            output = self._settings.chunk_separator.join(
                self._format_hit(hit) for hit in filtered
            )
            log_tool_end("search_documents", output)
            return output
        except Exception as exc:
            log_error("search_documents", exc)
            output = f"RETRIEVAL_ERROR: {exc}"
            log_tool_end("search_documents", output)
            return output

    @staticmethod
    def _format_hit(hit: SearchResult) -> str:
        filename = hit.metadata.get("filename", "unknown")
        return (
            f"Document ID: {hit.document_id}\n"
            f"File Name: {filename}\n"
            f"Score: {hit.score:.4f}\n"
            f"Content: {hit.document.strip()}"
        )

    def create_tools(self) -> list[BaseTool]:
        """Return the list of tools bound for the agent graph."""
        search_tool = tool("search_documents")(self._search_documents)
        return [search_tool]
