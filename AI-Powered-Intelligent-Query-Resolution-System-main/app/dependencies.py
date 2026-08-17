"""Shared FastAPI dependencies."""

from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.services.query_service import get_query_service

__all__ = ["get_current_user", "get_db", "get_query_service"]
