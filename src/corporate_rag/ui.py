from __future__ import annotations

import gradio as gr

from corporate_rag.citations import citation
from corporate_rag.config import settings
from corporate_rag.ingestion import load_source_chunks
from corporate_rag.runtime import create_assistant

LOADING_INDICATOR = """
<div class=\"answer-loading\" role=\"status\" aria-live=\"polite\">
  <span class=\"answer-spinner\"></span> Ищу информацию и готовлю ответ…
</div>
"""

LOADING_CSS = """
.answer-loading { align-items: center; color: var(--body-text-color); display: flex; gap: 8px; min-height: 28px; }
.answer-spinner { animation: answer-spin .8s linear infinite; border: 3px solid var(--border-color-primary); border-radius: 50%; border-top-color: var(--color-accent); height: 16px; width: 16px; }
@keyframes answer-spin { to { transform: rotate(360deg); } }
"""


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
        if not question.strip():
            yield "Введите вопрос.", "", gr.update(value="", visible=False)
            return

        # Clear the previous result before the synchronous RAG call begins.
        yield "", "", gr.update(value=LOADING_INDICATOR, visible=True)
        result = assistant.answer(question, history=history)
        history.append((question, result.text))
        source_text = "\n".join(
            f"- {citation(item.metadata)} — {item.metadata.get('header', 'без заголовка')} "
            f"(score: {item.score:.3f})"
            for item in result.sources
        ) or "Нет источников: ответ сформирован как отказ."
        yield result.text, source_text, gr.update(value="", visible=False)

    with gr.Blocks(title="Corporate RAG Assistant", css=LOADING_CSS) as app:
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
                loading = gr.HTML(visible=False)

        source_document.change(show_document, inputs=source_document, outputs=document_text)
        ask_button.click(
            ask,
            inputs=question,
            outputs=[answer, sources, loading],
            show_progress="hidden",
        )
    return app


def main() -> None:
    build_ui().launch(server_name=settings.gradio_host, server_port=settings.gradio_port)
