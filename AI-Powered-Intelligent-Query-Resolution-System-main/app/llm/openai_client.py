"""OpenAI GPT chat LLM factory."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.llm.exceptions import LLMConfigError


def create_openai_chat_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI client from application settings.

    Raises:
        LLMConfigError: If ``OPENAI_API_KEY`` is missing.
    """
    cfg = settings or get_settings()
    if not cfg.openai_api_key.strip():
        raise LLMConfigError(
            "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
            "Set it in your .env file."
        )
    return ChatOpenAI(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=cfg.llm_temperature,
    )
