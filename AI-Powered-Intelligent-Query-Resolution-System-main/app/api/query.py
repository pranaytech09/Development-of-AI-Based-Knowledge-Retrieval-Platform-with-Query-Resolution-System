"""Query resolution API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.schemas.auth import UserPublic
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import QueryService, get_query_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def ask_question(
    body: QueryRequest,
    _user: UserPublic = Depends(get_current_user),
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Ask a question; returns answer plus source chunks, confidence, and citations."""
    try:
        return service.ask(body.question, thread_id=body.thread_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc


@router.post("/reset", response_model=dict[str, str])
async def reset_conversation(
    _user: UserPublic = Depends(get_current_user),
    service: QueryService = Depends(get_query_service),
) -> dict[str, str]:
    """Clear the current conversation thread and start a new one."""
    thread_id = service.reset_thread()
    return {"thread_id": thread_id, "message": "Conversation reset."}
