"""Provider-agnostic chat LLM factory."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.llm.exceptions import LLMConfigError
from app.llm.ollama_client import create_ollama_chat_llm
from app.llm.openai_client import create_openai_chat_llm


def create_chat_llm(settings: Settings | None = None) -> Any:
    """Return a LangChain chat model for the configured provider.

    Supported ``LLM_PROVIDER`` values:
    - ``ollama`` — local ChatOllama (default)
    - ``openai`` — OpenAI GPT via ChatOpenAI

    Raises:
        LLMConfigError: If the provider is unknown or required settings are missing.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider.strip().lower()

    if provider == "ollama":
        return create_ollama_chat_llm(cfg)
    if provider == "openai":
        return create_openai_chat_llm(cfg)

    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER={cfg.llm_provider!r}. "
        "Use 'ollama' or 'openai'."
    )
