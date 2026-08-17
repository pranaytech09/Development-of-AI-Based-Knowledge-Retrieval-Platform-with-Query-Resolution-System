"""Conditional edge routers for the agent graphs."""

from __future__ import annotations

from typing import Any, Literal, Union

from langgraph.types import Send

from app.agents.execution_logger import log_route
from app.agents.state import AgentState, State
from app.core.config import get_settings


def route_after_rewrite(
    state: State,
) -> Union[Literal["request_clarification"], list[Send]]:
    """Route to clarification interrupt or fan-out agent subgraphs."""
    if not state.get("questionIsClear", False):
        decision: Any = "request_clarification"
    else:
        decision = [
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
    log_route("after_rewrite", decision, state)
    return decision


def route_after_orchestrator_call(
    state: AgentState,
) -> Literal["tools", "fallback_response", "collect_answer"]:
    """Decide whether to run tools, fall back, or collect the final answer."""
    settings = get_settings()
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        decision: Literal["tools", "fallback_response", "collect_answer"] = "collect_answer"
        log_route("after_orchestrator_call", decision, state)
        return decision

    if iteration >= settings.max_iterations or tool_count > settings.max_tool_calls:
        decision = "fallback_response"
        log_route("after_orchestrator_call", decision, state)
        return decision

    decision = "tools"
    log_route("after_orchestrator_call", decision, state)
    return decision
