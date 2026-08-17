"""Temporary Gradio frontend for document upload + chat.

Replace with React (Module 8) when the query API is stable.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr

from app.services.query_service import QueryService, get_query_service
from app.services.upload_service import UploadService


def create_gradio_ui(
    query_service: QueryService | None = None,
    upload_service: UploadService | None = None,
) -> gr.Blocks:
    """Build a temporary Gradio Blocks app for local demos."""
    qs = query_service or get_query_service()
    us = upload_service or UploadService()

    def format_file_list() -> str:
        files = us.list_documents()
        if not files:
            return "No documents in the knowledge base."
        return "\n".join(files)

    def upload_handler(files: list[str] | None, progress: gr.Progress = gr.Progress()) -> tuple[None, str]:
        if not files:
            return None, format_file_list()
        paths = [Path(f) for f in files]
        progress(0.2, desc="Saving and indexing…")
        added, chunks = us.save_and_ingest(paths)
        progress(1.0, desc="Done")
        gr.Info(f"Added: {len(added)} file(s) | Indexed chunks: {chunks}")
        return None, format_file_list()

    def clear_handler() -> str:
        try:
            us.clear_all()
            gr.Info("Removed all documents")
        except Exception as exc:
            gr.Error(f"Unable to clear documents: {exc}")
        return format_file_list()

    def chat_handler(msg: str, _hist: list) -> Generator[list[dict[str, Any]], None, None]:
        for chunk in qs.stream_chat(msg):
            yield chunk

    def clear_chat_handler() -> None:
        qs.reset_thread()

    with gr.Blocks(title="AI Query Resolution (temp Gradio)") as demo:
        gr.Markdown(
            "## AI-Powered Intelligent Query Resolution\n"
            "_Temporary Gradio UI — will be replaced by React._"
        )

        with gr.Tab("Documents"):
            gr.Markdown("Upload PDF or Word (`.docx`) files, then index into ChromaDB.")
            files_input = gr.File(
                label="Drop PDF or DOCX files here",
                file_count="multiple",
                type="filepath",
                height=180,
            )
            add_btn = gr.Button("Add Documents", variant="primary")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines=8,
                label="Current documents",
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear All", variant="stop")

            add_btn.click(
                upload_handler,
                [files_input],
                [files_input, file_list],
                show_progress="full",
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=640,
                placeholder="<strong>Ask about your documents</strong>",
                show_label=False,
                layout="bubble",
            )
            chatbot.clear(clear_chat_handler)
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot)

    return demo


def launch_gradio(*, share: bool = False, server_port: int = 7860) -> None:
    """Initialize services and launch Gradio on its own port."""
    get_query_service()
    demo = create_gradio_ui()
    demo.queue().launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch_gradio()
