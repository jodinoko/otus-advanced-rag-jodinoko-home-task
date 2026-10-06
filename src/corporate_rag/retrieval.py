from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from corporate_rag.config import settings
from corporate_rag.models import RetrievedChunk


def _filters(values: dict[str, Any] | None) -> MetadataFilters | None:
    if not values:
        return None
    return MetadataFilters(filters=[ExactMatchFilter(key=key, value=value) for key, value in values.items()])


def _to_chunk(node) -> RetrievedChunk:
    return RetrievedChunk(
        text=node.node.get_content(metadata_mode="none"),
        metadata=dict(node.node.metadata),
        score=float(node.score or 0.0),
    )


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, str, str, str]:
    """Identify a source fragment independently of its vector-store node ID."""
    return (
        str(chunk.metadata.get("source", "")),
        str(chunk.metadata.get("page", "")),
        str(chunk.metadata.get("header", "")),
        chunk.text,
    )


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the highest-ranked occurrence of each identical source fragment."""
    unique: list[RetrievedChunk] = []
    seen: set[tuple[str, str, str, str]] = set()
    for chunk in chunks:
        key = _chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


class HybridRetriever:
    """Dense + BM25 retrieval combined with reciprocal-rank fusion."""

    def __init__(self, index: VectorStoreIndex, nodes: list):
        self.index = index
        self.nodes = nodes
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=settings.retrieval_top_k)

    def retrieve(self, query: str, metadata: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        dense = self.index.as_retriever(
            similarity_top_k=settings.retrieval_top_k, filters=_filters(metadata)
        ).retrieve(query)
        sparse = self.bm25.retrieve(query)
        if metadata:
            sparse = [
                item for item in sparse
                if all(str(item.node.metadata.get(key)) == str(value) for key, value in metadata.items())
            ]
        fused: dict[str, tuple[Any, float]] = {}
        for ranking in (dense, sparse):
            for rank, item in enumerate(ranking, start=1):
                key = item.node.node_id
                previous_item, previous_score = fused.get(key, (item, 0.0))
                fused[key] = (previous_item, previous_score + 1.0 / (60 + rank))
        ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        return [
            RetrievedChunk(
                text=item.node.get_content(metadata_mode="none"),
                metadata=dict(item.node.metadata),
                score=score,
            )
            for item, score in ranked[: settings.retrieval_top_k]
        ]


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(settings.rerank_model)
        scores = self._model.predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda row: float(row[1]), reverse=True)
        rescored = [
            RetrievedChunk(text=chunk.text, metadata=chunk.metadata, score=float(score))
            for chunk, score in ranked
        ]
        return deduplicate_chunks(rescored)[: settings.rerank_top_n]
