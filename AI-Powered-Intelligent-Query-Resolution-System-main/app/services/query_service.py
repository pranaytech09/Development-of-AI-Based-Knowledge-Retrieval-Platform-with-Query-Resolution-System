"""Query service: bootstrap LangGraph and run grounded Q&A."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Generator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.agents.execution_logger import log_chat_end, log_chat_start, log_error
from app.agents.graph import create_agent_graph
from app.agents.tools import ToolFactory
from app.core.config import Settings, get_settings
from app.llm import create_chat_llm
from app.rag.retriever import DocumentRetriever
from app.schemas.query import QueryResponse, SourceChunkPublic
from app.transparency.citations import extract_citations, extract_source_chunks
from app.transparency.confidence import compute_retrieval_confidence

SYSTEM_NODES = {"summarize_history", "rewrite_query"}
FINAL_RESPONSE_NODES = {"aggregate_answers"}


@dataclass
class StreamMessage:
    """A Gradio-friendly assistant message chunk."""

    role: str = "assistant"
    content: str = ""
    title: str | None = None
    node: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        metadata = {k: v for k, v in {"title": self.title, "node": self.node}.items() if v}
        if metadata:
            payload["metadata"] = metadata
        return payload


@dataclass
class QueryService:
    """Owns the compiled agent graph and session thread id."""

    settings: Settings = field(default_factory=get_settings)
    retriever: DocumentRetriever | None = None
    agent_graph: Any = field(default=None, init=False)
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def initialize(self) -> None:
        """Compile the LangGraph agent with DocumentRetriever-backed tools."""
        llm = create_chat_llm(self.settings)
        retriever = self.retriever or DocumentRetriever()
        tools = ToolFactory(retriever=retriever, settings=self.settings).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_run_config(self, thread_id: str | None = None) -> dict[str, Any]:
        """Build LangGraph invoke/stream config for a conversation thread."""
        tid = thread_id or self.thread_id
        return {
            "configurable": {"thread_id": tid},
            "recursion_limit": self.settings.graph_recursion_limit,
        }

    def reset_thread(self) -> str:
        """Start a fresh conversation thread and return the new id."""
        if self.agent_graph is not None:
            try:
                self.agent_graph.checkpointer.delete_thread(self.thread_id)
            except Exception:
                pass
        self.thread_id = str(uuid.uuid4())
        return self.thread_id

    def ask(self, question: str, thread_id: str | None = None) -> QueryResponse:
        """Run one turn (or resume after clarification) and return a transparent answer."""
        if self.agent_graph is None:
            raise RuntimeError("QueryService is not initialized. Call initialize() first.")

        tid = thread_id or self.thread_id
        config = self.get_run_config(tid)
        current_state = self.agent_graph.get_state(config)
        log_chat_start(question.strip(), tid, bool(current_state.next))

        try:
            if current_state.next:
                self.agent_graph.update_state(
                    config, {"messages": [HumanMessage(content=question.strip())]}
                )
                self.agent_graph.invoke(None, config=config)
            else:
                self.agent_graph.invoke(
                    {"messages": [HumanMessage(content=question.strip())]},
                    config=config,
                )
        except Exception as exc:
            log_error("ask", exc)
            raise

        final_state = self.agent_graph.get_state(config)
        values = getattr(final_state, "values", {}) or {}
        log_chat_end(values)
        return self._build_response(values, tid, pending_interrupt=bool(final_state.next))

    def stream_chat(
        self, message: str, thread_id: str | None = None
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Yield Gradio chatbot message lists while the graph streams."""
        if self.agent_graph is None:
            yield [{"role": "assistant", "content": "System not initialized."}]
            return

        tid = thread_id or self.thread_id
        config = self.get_run_config(tid)
        current_state = self.agent_graph.get_state(config)
        log_chat_start(message.strip(), tid, bool(current_state.next))

        try:
            if current_state.next:
                self.agent_graph.update_state(
                    config, {"messages": [HumanMessage(content=message.strip())]}
                )
                stream_input = None
            else:
                stream_input = {"messages": [HumanMessage(content=message.strip())]}

            response_messages: list[dict[str, Any]] = []
            active_tool_calls: dict[str, int] = {}

            for chunk, metadata in self.agent_graph.stream(
                stream_input, config=config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")

                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if tc.get("id") and tc["id"] not in active_tool_calls:
                            response_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Running `{tc['name']}`...",
                                    "metadata": {"title": tc["name"]},
                                }
                            )
                            active_tool_calls[tc["id"]] = len(response_messages) - 1

                elif isinstance(chunk, ToolMessage):
                    idx = active_tool_calls.get(chunk.tool_call_id)
                    if idx is not None:
                        preview = str(chunk.content)[:300]
                        suffix = "\n..." if len(str(chunk.content)) > 300 else ""
                        response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

                elif isinstance(chunk, AIMessageChunk) and chunk.content and node in FINAL_RESPONSE_NODES:
                    last = response_messages[-1] if response_messages else None
                    if not (last and last.get("role") == "assistant" and "metadata" not in last):
                        response_messages.append({"role": "assistant", "content": ""})
                    response_messages[-1]["content"] += chunk.content

                elif node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    # Surface clarification text when rewrite emits it as named AI message later
                    pass

                else:
                    continue

                yield response_messages

            final_state = self.agent_graph.get_state(config)
            values = getattr(final_state, "values", {}) or {}
            log_chat_end(values)

            # Ensure clarification messages appear if the graph interrupted
            if final_state.next and not any(
                m.get("content") for m in response_messages if "metadata" not in m
            ):
                clarification = self._last_clarification(values)
                if clarification:
                    yield [{"role": "assistant", "content": clarification}]

        except Exception as exc:
            log_error("stream_chat", exc)
            yield [{"role": "assistant", "content": f"Error: {exc}"}]

    def _build_response(
        self,
        values: dict[str, Any],
        thread_id: str,
        *,
        pending_interrupt: bool,
    ) -> QueryResponse:
        messages = values.get("messages", []) or []
        clarification_needed = pending_interrupt or bool(values.get("pendingQuery"))

        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                if getattr(msg, "name", None) in {"agent_response", "clarification_response"}:
                    continue
                answer = str(msg.content)
                break

        contexts: list[str] = []
        for item in values.get("agent_answers", []) or []:
            if isinstance(item, dict):
                contexts.extend(item.get("contexts") or [])

        chunks = extract_source_chunks(contexts)
        return QueryResponse(
            answer=answer or ("Please clarify your question." if clarification_needed else ""),
            thread_id=thread_id,
            clarification_needed=clarification_needed,
            confidence=compute_retrieval_confidence(contexts),
            citations=extract_citations(contexts),
            source_chunks=[
                SourceChunkPublic(
                    document_id=c.document_id,
                    filename=c.filename,
                    score=c.score,
                    content=c.content,
                )
                for c in chunks
            ],
        )

    @staticmethod
    def _last_clarification(values: dict[str, Any]) -> str:
        for msg in reversed(values.get("messages", []) or []):
            if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "clarification":
                return str(msg.content)
        return ""


# Process-wide singleton used by API + Gradio
_query_service: QueryService | None = None


def get_query_service() -> QueryService:
    """Return the shared QueryService, initializing on first use."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
        _query_service.initialize()
    return _query_service
