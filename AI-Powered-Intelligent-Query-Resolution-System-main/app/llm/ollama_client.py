"""Ollama chat LLM factory."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import Settings, get_settings


def create_ollama_chat_llm(settings: Settings | None = None) -> ChatOllama:
    """Build a ChatOllama client from application settings."""
    cfg = settings or get_settings()
    return ChatOllama(
        model=cfg.llm_model,
        base_url=cfg.ollama_base_url,
        temperature=cfg.llm_temperature,
        seed=cfg.llm_seed,
        num_ctx=cfg.llm_num_ctx,
    )
