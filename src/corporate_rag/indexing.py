from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from corporate_rag.config import settings
from corporate_rag.ingestion import build_nodes, load_source_chunks


def configure_embeddings() -> None:
    LlamaSettings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        cache_folder=str(settings.embedding_cache_dir),
    )


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def vector_store() -> QdrantVectorStore:
    client = qdrant_client()
    return QdrantVectorStore(client=client, collection_name=settings.collection_name)


def build_index(reset: bool = False) -> tuple[VectorStoreIndex, list]:
    configure_embeddings()
    client = qdrant_client()
    if reset and client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    nodes = build_nodes(load_source_chunks())
    store = vector_store()
    index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))
    return index, nodes


def load_index() -> VectorStoreIndex:
    configure_embeddings()
    return VectorStoreIndex.from_vector_store(vector_store())
