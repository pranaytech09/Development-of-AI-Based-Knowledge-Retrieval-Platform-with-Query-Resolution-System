"""Unit tests for agent graph edge routers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langgraph.types import Send

from app.agents.edges import route_after_orchestrator_call, route_after_rewrite


def test_route_after_rewrite_clarification() -> None:
    state = {"questionIsClear": False, "rewrittenQuestions": []}
    assert route_after_rewrite(state) == "request_clarification"


def test_route_after_rewrite_fans_out() -> None:
    state = {
        "questionIsClear": True,
        "rewrittenQuestions": ["What is PTO?", "How is PTO accrued?"],
    }
    result = route_after_rewrite(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, Send) for item in result)
    assert result[0].node == "agent"
    assert result[0].arg["question"] == "What is PTO?"


def test_route_after_orchestrator_collects_when_no_tools() -> None:
    msg = SimpleNamespace(tool_calls=None)
    state = {"messages": [msg], "iteration_count": 1, "tool_call_count": 0}
    assert route_after_orchestrator_call(state) == "collect_answer"


def test_route_after_orchestrator_uses_tools() -> None:
    msg = SimpleNamespace(tool_calls=[{"name": "search_documents", "args": {"query": "pto"}}])
    state = {"messages": [msg], "iteration_count": 1, "tool_call_count": 1}
    with patch("app.agents.edges.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_iterations=10, max_tool_calls=8)
        assert route_after_orchestrator_call(state) == "tools"


def test_route_after_orchestrator_fallback_on_budget() -> None:
    msg = SimpleNamespace(tool_calls=[{"name": "search_documents", "args": {"query": "pto"}}])
    state = {"messages": [msg], "iteration_count": 99, "tool_call_count": 99}
    with patch("app.agents.edges.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_iterations=10, max_tool_calls=8)
        assert route_after_orchestrator_call(state) == "fallback_response"
