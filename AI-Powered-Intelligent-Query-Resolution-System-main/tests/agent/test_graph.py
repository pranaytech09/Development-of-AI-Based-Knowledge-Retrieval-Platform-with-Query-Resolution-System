"""Smoke tests for graph compilation with mocked tools."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.agents.graph import create_agent_graph


def test_create_agent_graph_compiles() -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    tools: list = []
    graph = create_agent_graph(llm, tools)
    assert graph is not None
    llm.bind_tools.assert_called_once_with(tools)
