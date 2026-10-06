from __future__ import annotations

import gradio as gr

from corporate_rag.citations import citation
from corporate_rag.config import settings
from corporate_rag.ingestion import load_source_chunks
from corporate_rag.runtime import create_assistant


def document_catalog() -> dict[str, str]:
    """Prepare source documents for the read-only panel in the UI."""
    documents: dict[str, list[str]] = {}
    for chunk in load_source_chunks():
        source = str(chunk.metadata["source"])
        location = (
            f"Страница {chunk.metadata['page']}"
            if chunk.metadata.get("page")
            else str(chunk.metadata.get("header", "Документ"))
        )
        documents.setdefault(source, []).append(f"### {location}\n\n{chunk.text}")
    return {source: "\n\n---\n\n".join(parts) for source, parts in sorted(documents.items())}


def build_ui() -> gr.Blocks:
    assistant = create_assistant()
    history: list[tuple[str, str]] = []
    catalog = document_catalog()
    document_names = list(catalog)

    def show_document(source: str) -> str:
        return catalog.get(source, "Выберите документ.")

    def ask(question: str):
        result = assistant.answer(question, history=history)
        history.append((question, result.text))
        source_text = "\n".join(
            f"- {citation(item.metadata)} — {item.metadata.get('header', 'без заголовка')} "
            f"(score: {item.score:.3f})"
            for item in result.sources
        ) or "Нет источников: ответ сформирован как отказ."
        return result.text, source_text

    with gr.Blocks(title="Corporate RAG Assistant") as app:
        gr.Markdown("# Корпоративный ассистент\nОтвечает только по базе знаний и показывает источники.")
        gr.Markdown("## Документы базы знаний")
        with gr.Row():
            with gr.Column(scale=1):
                source_document = gr.Dropdown(
                    document_names,
                    label="Документ",
                    value=document_names[0] if document_names else None,
                )
                document_text = gr.Markdown(
                    show_document(document_names[0]) if document_names else "Документы не найдены."
                )
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Вопрос",
                    placeholder="Как получить доступ к корпоративному VPN?",
                )
                ask_button = gr.Button("Спросить", variant="primary")
                answer = gr.Markdown(label="Ответ")
                sources = gr.Markdown(label="Извлечённые источники")

        source_document.change(show_document, inputs=source_document, outputs=document_text)
        ask_button.click(
            ask,
            inputs=question,
            outputs=[answer, sources],
            show_progress="minimal",
            show_progress_on=answer,
        )
    return app


def main() -> None:
    build_ui().launch(server_name=settings.gradio_host, server_port=settings.gradio_port)
