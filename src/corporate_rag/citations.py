from __future__ import annotations

from corporate_rag.models import RetrievedChunk


def citation(metadata: dict) -> str:
    source = metadata.get("source", "Неизвестный источник")
    page = metadata.get("page")
    return f"[{source}, стр. {page}]" if page else f"[{source}]"


def context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"Контекст {number} (Источник: {citation(chunk.metadata)}; раздел: "
            f"{chunk.metadata.get('header', 'без заголовка')}):\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
