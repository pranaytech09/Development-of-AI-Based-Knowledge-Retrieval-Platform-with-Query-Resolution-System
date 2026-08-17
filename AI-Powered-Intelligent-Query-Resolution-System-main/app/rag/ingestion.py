"""Ingest uploaded PDF and Word documents into the vector store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from app.schemas.rag import DocumentChunk
from app.rag.config import RAGConfig
from app.rag.document_loader import chunk_text, extract_text, list_uploaded_files
from app.rag.embedding_service import EmbeddingService, OllamaEmbeddingService
from app.rag.vector_store import ChromaVectorStore, VectorStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


class DocumentSource(Protocol):
    """Any source that yields document chunks for indexing."""

    def load_chunks(self) -> list[DocumentChunk]: ...


class UploadsDocumentSource:
    """Loads and chunks files from the uploads directory."""

    def __init__(self, uploads_dir: Path | None = None) -> None:
        self._uploads_dir = uploads_dir or RAGConfig().uploads_dir

    def load_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for path in list_uploaded_files(self._uploads_dir):
            try:
                text = extract_text(path)
                for index, content in enumerate(chunk_text(text)):
                    chunks.append(
                        DocumentChunk(
                            filename=path.name,
                            file_type=path.suffix.lower().lstrip("."),
                            chunk_index=index,
                            content=content,
                            source_path=str(path),
                        )
                    )
            except Exception:
                logger.exception("Failed to process %s", path.name)
        return chunks


class IngestionPipeline:
    """Loads documents, embeds them, and persists to ChromaDB."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        document_source: DocumentSource | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._document_source = document_source or UploadsDocumentSource(self._config.uploads_dir)
        self._embeddings = embedding_service or OllamaEmbeddingService(self._config)
        self._store = vector_store or ChromaVectorStore(self._config)

    def run(self, *, reset: bool = False) -> int:
        """Ingest all uploaded documents. Returns number of chunks indexed."""
        chunks = self._document_source.load_chunks()
        if not chunks:
            logger.warning("No documents found in %s", self._config.uploads_dir)
            return 0

        if reset:
            self._store.reset()

        indexed = 0
        for batch in _chunk(chunks, BATCH_SIZE):
            ids = [chunk.document_id for chunk in batch]
            documents = [chunk.to_document_text() for chunk in batch]
            metadatas = [chunk.to_metadata() for chunk in batch]
            embeddings = self._embeddings.embed_documents(documents)
            self._store.upsert(ids, embeddings, documents, metadatas)
            indexed += len(batch)
            logger.info("Indexed %s / %s chunks", indexed, len(chunks))

        return indexed


def _chunk(items: list[DocumentChunk], size: int) -> list[list[DocumentChunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def ingest_documents(*, reset: bool = False) -> int:
    """Convenience entrypoint for CLI or scripts."""
    logging.basicConfig(level=logging.INFO)
    return IngestionPipeline().run(reset=reset)


if __name__ == "__main__":
    count = ingest_documents(reset=True)
    print(f"Ingested {count} document chunks into ChromaDB.")

