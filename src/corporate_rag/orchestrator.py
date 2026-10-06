from __future__ import annotations

from dataclasses import dataclass

from corporate_rag.citations import context_block
from corporate_rag.config import settings
from corporate_rag.llm import complete
from corporate_rag.models import Answer
from corporate_rag.observability import trace_answer
from corporate_rag.prompts import SYSTEM_PROMPT, user_prompt
from corporate_rag.retrieval import CrossEncoderReranker, HybridRetriever


@dataclass
class CorporateAssistant:
    retriever: HybridRetriever
    reranker: CrossEncoderReranker

    def answer(
        self,
        question: str,
        metadata_filters: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        clean_question = question.strip()
        if not clean_question:
            return Answer(text="Я не знаю", refused=True)

        with trace_answer(clean_question, metadata_filters) as trace:
            candidates = self.retriever.retrieve(clean_question, metadata_filters)
            sources = self.reranker.rerank(clean_question, candidates)
            if not sources or sources[0].score < settings.min_rerank_score:
                result = Answer(text="Я не знаю", refused=True)
            else:
                # A compact history supports follow-ups but retrieved evidence remains authoritative.
                recent = (history or [])[-3:]
                history_text = "\n".join(f"Пользователь: {q}\nАссистент: {a}" for q, a in recent)
                prompt = user_prompt(clean_question, context_block(sources))
                if history_text:
                    prompt = f"Недавний диалог (не является источником фактов):\n{history_text}\n\n{prompt}"
                response = complete(SYSTEM_PROMPT, prompt) or "Я не знаю"
                result = Answer(text=response, sources=sources, refused=response == "Я не знаю")
            if trace is not None:
                trace.update(
                    output={"answer": result.text, "sources": [item.metadata for item in result.sources]},
                    metadata={"refused": result.refused, "retrieved_count": len(result.sources)},
                )
            return result
