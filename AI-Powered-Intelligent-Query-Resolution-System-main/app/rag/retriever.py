"""Semantic search over indexed documents."""

from __future__ import annotations

from typing import Any

from app.schemas.rag import SearchResult
from app.rag.config import RAGConfig
from app.rag.embedding_service import EmbeddingService, OllamaEmbeddingService
from app.rag.vector_store import ChromaVectorStore, VectorStore


class DocumentRetriever:
    """Embeds queries and retrieves matching document chunks."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._embeddings = embedding_service or OllamaEmbeddingService(self._config)
        self._store = vector_store or ChromaVectorStore(self._config)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return top-k document chunks semantically similar to the query."""
        k = top_k or self._config.default_top_k
        query_vector = self._embeddings.embed_query(query)
        raw = self._store.similarity_search(query_vector, k, where=filters)
        return self._parse_results(raw)

    def _parse_results(self, raw: dict[str, Any]) -> list[SearchResult]:
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        return [
            SearchResult(
                document_id=doc_id,
                score=1.0 - float(distance),
                document=doc or "",
                metadata={k: str(v) for k, v in (meta or {}).items()},
            )
            for doc_id, doc, meta, distance in zip(ids, documents, metadatas, distances)
        ]

