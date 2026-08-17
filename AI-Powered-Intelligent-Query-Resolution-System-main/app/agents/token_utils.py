"""Token estimation helpers for context compression decisions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _get_token_encoding() -> Any:
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model("gpt-4")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_context_tokens(messages: list[Any]) -> int:
    """Estimate token count for message contents (tiktoken with char fallback)."""
    contents = [
        str(msg.content) for msg in messages if hasattr(msg, "content") and msg.content
    ]
    encoding = _get_token_encoding()
    if encoding is None:
        return sum(max(1, len(content) // 4) for content in contents)
    return sum(len(encoding.encode(content)) for content in contents)
