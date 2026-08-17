"""Document upload and ingestion API."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.auth.dependencies import get_current_user
from app.rag.document_loader import SUPPORTED_EXTENSIONS
from app.schemas.auth import UserPublic
from app.schemas.query import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["upload"])


def get_upload_service() -> UploadService:
    return UploadService()


@router.get("/documents", response_model=list[str])
async def list_documents(
    _user: UserPublic = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
) -> list[str]:
    """List filenames currently available for retrieval."""
    return service.list_documents()


@router.post("", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    _user: UserPublic = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    """Accept PDF/DOCX uploads, save to uploads/, and run ingestion."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    saved_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for upload in files:
            name = upload.filename or "upload.bin"
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {suffix}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
                )
            dest = tmp_dir / name
            dest.write_bytes(await upload.read())
            saved_paths.append(dest)

        filenames, chunks = service.save_and_ingest(saved_paths)

    return UploadResponse(
        filenames=filenames,
        chunks_indexed=chunks,
        message=f"Indexed {chunks} chunks from {len(filenames)} file(s).",
    )


@router.delete("", response_model=UploadResponse)
async def clear_documents(
    _user: UserPublic = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    """Remove all uploaded documents and reset the vector store."""
    service.clear_all()
    return UploadResponse(filenames=[], chunks_indexed=0, message="Knowledge base cleared.")
