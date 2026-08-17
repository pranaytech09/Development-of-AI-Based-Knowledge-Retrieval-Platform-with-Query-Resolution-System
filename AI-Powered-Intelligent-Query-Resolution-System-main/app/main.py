"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth_router, query_router, upload_router
from app.core.config import get_settings
from app.database.connection import init_db
from app.services.query_service import get_query_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create DB tables and warm the query agent on startup."""
    settings = get_settings()
    try:
        await init_db()
    except Exception as exc:
        if settings.debug:
            print(f"Warning: Database initialization failed (PostgreSQL not running?): {exc}")
    try:
        get_query_service()
    except Exception as exc:
        if settings.debug:
            print(f"Warning: QueryService failed to initialize: {exc}")
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(query_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
