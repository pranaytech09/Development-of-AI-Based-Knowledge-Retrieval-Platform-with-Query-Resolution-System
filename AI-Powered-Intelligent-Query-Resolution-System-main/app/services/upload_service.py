"""Document upload and ingestion service."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.rag.config import RAGConfig
from app.rag.document_loader import SUPPORTED_EXTENSIONS, list_uploaded_files
from app.rag.ingestion import IngestionPipeline
from app.rag.vector_store import ChromaVectorStore, VectorStore


class UploadService:
    """Save uploaded files and run the existing ingestion pipeline."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        pipeline: IngestionPipeline | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        # One store shared with the pipeline: two handles to the same collection
        # go stale as soon as either one resets it.
        self._store = vector_store or ChromaVectorStore(self._config)
        self._pipeline = pipeline or IngestionPipeline(
            config=self._config, vector_store=self._store
        )
        self._config.uploads_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[str]:
        """Return filenames currently in the uploads directory."""
        return [path.name for path in list_uploaded_files(self._config.uploads_dir)]

    def save_and_ingest(self, file_paths: list[Path], *, reset: bool = False) -> tuple[list[str], int]:
        """Copy files into uploads and re-index. Returns (saved names, chunk count)."""
        saved: list[str] = []
        for src in file_paths:
            suffix = src.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            dest = self._config.uploads_dir / src.name
            shutil.copy2(src, dest)
            saved.append(dest.name)

        if not saved and not reset:
            return [], 0

        chunks = self._pipeline.run(reset=reset)
        return saved, chunks

    def clear_all(self) -> None:
        """Delete uploaded files and reset the vector collection."""
        for path in list_uploaded_files(self._config.uploads_dir):
            path.unlink(missing_ok=True)
        self._store.reset()
