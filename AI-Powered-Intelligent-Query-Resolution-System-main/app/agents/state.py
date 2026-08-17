"""Shared LangGraph state for the main graph and agent subgraph."""

from __future__ import annotations

from typing import Annotated, List, Set

import operator
from langgraph.graph import MessagesState


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """Append agent answers, or clear the list when a reset sentinel appears."""
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """Merge retrieval key sets."""
    return a | b


def append_unique(existing: List[str], new: List[str]) -> List[str]:
    """Append strings while preserving order and uniqueness."""
    return list(dict.fromkeys(existing + new))


class State(MessagesState):
    """State for the outer (main) agent graph."""

    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    pendingQuery: str = ""
    pendingClarifications: List[str] = []
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


class AgentState(MessagesState):
    """State for the per-question agent subgraph."""

    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    retrieved_contexts: Annotated[List[str], append_unique] = []
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
