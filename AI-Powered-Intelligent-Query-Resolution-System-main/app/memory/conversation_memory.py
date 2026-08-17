"""Conversation memory package.

Multi-turn memory currently lives inside the LangGraph state
(``conversation_summary`` + checkpointer thread). This module is a thin
facade for future extraction into Module 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """In-memory rolling summary placeholder for a chat session."""

    thread_id: str
    summary: str = ""
    pending_query: str = ""
    pending_clarifications: list[str] = field(default_factory=list)

    def clear(self) -> None:
        """Reset session memory fields."""
        self.summary = ""
        self.pending_query = ""
        self.pending_clarifications = []
