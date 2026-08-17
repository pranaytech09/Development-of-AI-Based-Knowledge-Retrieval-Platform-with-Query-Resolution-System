"""Unit tests for Chroma client selection (no Chroma server required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromadb.errors import NotFoundError

from app.rag.config import EMBEDDED_MODE, HTTP_MODE, RAGConfig
from app.rag.exceptions import VectorStoreConnectionError
from app.rag.vector_store import ChromaVectorStore


@pytest.fixture
def http_config() -> RAGConfig:
    return RAGConfig(chroma_mode=HTTP_MODE, chroma_host="localhost", chroma_port=6334)


class TestChromaVectorStoreClientSelection:
    """The configured mode decides which Chroma client is built."""

    def test_http_mode_builds_http_client(self, http_config: RAGConfig) -> None:
        with patch("app.rag.vector_store.chromadb") as chroma:
            ChromaVectorStore(http_config)

        chroma.HttpClient.assert_called_once_with(host="localhost", port=6334)
        chroma.PersistentClient.assert_not_called()

    def test_embedded_mode_builds_persistent_client(self, tmp_path: Path) -> None:
        config = RAGConfig(chroma_mode=EMBEDDED_MODE, persist_dir=tmp_path / "chromadb")

        with patch("app.rag.vector_store.chromadb") as chroma:
            ChromaVectorStore(config)

        chroma.PersistentClient.assert_called_once_with(path=str(config.persist_dir))
        chroma.HttpClient.assert_not_called()
        assert config.persist_dir.exists()

    def test_unreachable_server_raises_connection_error(self, http_config: RAGConfig) -> None:
        with patch("app.rag.vector_store.chromadb") as chroma:
            chroma.HttpClient.side_effect = ConnectionError("refused")

            with pytest.raises(VectorStoreConnectionError, match="localhost:6334"):
                ChromaVectorStore(http_config)

    def test_collection_created_with_cosine_space(self, http_config: RAGConfig) -> None:
        client = MagicMock()

        with patch("app.rag.vector_store.chromadb") as chroma:
            chroma.HttpClient.return_value = client
            ChromaVectorStore(http_config)

        client.get_or_create_collection.assert_called_once_with(
            name=http_config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class TestChromaVectorStoreStaleCollection:
    """A dropped-and-recreated collection invalidates the cached handle."""

    @staticmethod
    def _store(client: MagicMock, config: RAGConfig) -> ChromaVectorStore:
        with patch("app.rag.vector_store.chromadb") as chroma:
            chroma.HttpClient.return_value = client
            return ChromaVectorStore(config)

    def test_upsert_refreshes_stale_collection_and_retries(
        self, http_config: RAGConfig
    ) -> None:
        stale, fresh = MagicMock(), MagicMock()
        stale.upsert.side_effect = NotFoundError("Collection [abc] does not exist.")
        client = MagicMock()
        client.get_or_create_collection.side_effect = [stale, fresh]

        store = self._store(client, http_config)
        store.upsert(["id-1"], [[0.1]], ["text"], [{"filename": "a.pdf"}])

        fresh.upsert.assert_called_once_with(
            ids=["id-1"],
            embeddings=[[0.1]],
            documents=["text"],
            metadatas=[{"filename": "a.pdf"}],
        )

    def test_count_refreshes_stale_collection_and_retries(
        self, http_config: RAGConfig
    ) -> None:
        stale, fresh = MagicMock(), MagicMock()
        stale.count.side_effect = NotFoundError("Collection [abc] does not exist.")
        fresh.count.return_value = 7
        client = MagicMock()
        client.get_or_create_collection.side_effect = [stale, fresh]

        store = self._store(client, http_config)

        assert store.count() == 7

    def test_persistent_failure_propagates(self, http_config: RAGConfig) -> None:
        collection = MagicMock()
        collection.count.side_effect = NotFoundError("gone")
        client = MagicMock()
        client.get_or_create_collection.return_value = collection

        store = self._store(client, http_config)

        with pytest.raises(NotFoundError):
            store.count()

    def test_reset_tolerates_missing_collection(self, http_config: RAGConfig) -> None:
        client = MagicMock()
        client.delete_collection.side_effect = NotFoundError("already gone")

        store = self._store(client, http_config)
        store.reset()

        assert client.get_or_create_collection.call_count == 2
