"""ChromaDB vector store wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import chromadb
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from app.rag.config import HTTP_MODE, RAGConfig
from app.rag.exceptions import VectorStoreConnectionError

T = TypeVar("T")


class VectorStore(ABC):
    """Vector persistence contract."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        """Insert or update vectors."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return nearest neighbors for a query embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return stored document count."""

    @abstractmethod
    def reset(self) -> None:
        """Delete and recreate the collection."""


class ChromaVectorStore(VectorStore):
    """ChromaDB collection for document embeddings.

    Backed either by a standalone Chroma server or by an on-disk embedded
    store, selected through :class:`RAGConfig`.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        cfg = config or RAGConfig()
        self._collection_name = cfg.collection_name
        self._client = self._build_client(cfg)
        self._collection: Collection = self._create_collection()

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        self._with_collection(
            lambda collection: collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._with_collection(lambda collection: collection.query(**kwargs))

    def count(self) -> int:
        return self._with_collection(lambda collection: collection.count())

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except NotFoundError:
            pass
        self._collection = self._create_collection()

    def _with_collection(self, operation: Callable[[Collection], T]) -> T:
        """Run an operation, refreshing the cached handle if it went stale.

        The cached ``Collection`` holds a UUID that is invalidated whenever the
        collection is dropped and recreated — by ``reset()`` here or by another
        process sharing the same store.
        """
        try:
            return operation(self._collection)
        except NotFoundError:
            self._collection = self._create_collection()
            return operation(self._collection)

    def _create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_client(cfg: RAGConfig) -> ClientAPI:
        """Return a Chroma client for the configured mode.

        Raises:
            VectorStoreConnectionError: If the Chroma server is unreachable.
        """
        if cfg.chroma_mode != HTTP_MODE:
            cfg.persist_dir.mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=str(cfg.persist_dir))

        try:
            return chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
        except Exception as error:
            raise VectorStoreConnectionError(
                f"Cannot reach the Chroma server at {cfg.chroma_url}. "
                "Start it with Docker or set CHROMA_MODE=embedded."
            ) from error

