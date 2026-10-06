from __future__ import annotations

from corporate_rag.indexing import build_index, load_index
from corporate_rag.ingestion import build_nodes, load_source_chunks
from corporate_rag.orchestrator import CorporateAssistant
from corporate_rag.retrieval import CrossEncoderReranker, HybridRetriever


def create_assistant(rebuild: bool = False) -> CorporateAssistant:
    if rebuild:
        index, nodes = build_index(reset=True)
    else:
        nodes = build_nodes(load_source_chunks())
        index = load_index()
    return CorporateAssistant(HybridRetriever(index, nodes), CrossEncoderReranker())
