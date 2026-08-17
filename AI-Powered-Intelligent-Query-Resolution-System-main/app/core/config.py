"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai"]


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI-Powered Intelligent Query Resolution System"
    debug: bool = False

    # Default to a local SQLite async file for easy local testing. Set
    # `DATABASE_URL` in your environment to a PostgreSQL URL for production,
    # e.g. `postgresql+asyncpg://postgres:password@host:5432/dbname`.
    database_url: str = "sqlite+aiosqlite:///./ai_query_system.db"

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    access_cookie_name: str = "access_token"
    refresh_cookie_name: str = "refresh_token"

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Chat LLM provider: "ollama" (local) or "openai" (GPT)
    llm_provider: LLMProvider = "ollama"
    llm_temperature: float = 0.0

    # Ollama (used when llm_provider=ollama)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "granite4.1:8b"
    llm_seed: int = 42
    # Ollama allocates a KV cache sized to this window; lower it on low-RAM machines.
    llm_num_ctx: int = 4096

    # OpenAI GPT (used when llm_provider=openai)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Agentic retrieval loop
    retrieval_score_threshold: float = 0.3
    default_retrieval_k: int = 5
    chunk_separator: str = "\n\n<CHUNK_BOUNDARY>\n\n"
    max_tool_calls: int = 8
    max_iterations: int = 10
    graph_recursion_limit: int = 50
    main_history_messages_to_keep: int = 4
    base_token_threshold: int = 2000
    token_growth_factor: float = 0.9

    # Agent execution logging (stdout)
    execution_logging_enabled: bool = False
    execution_log_max_chars: int = 1200
    execution_log_use_color: bool = True

    # Temporary Gradio UI
    gradio_enabled: bool = True
    gradio_share: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
