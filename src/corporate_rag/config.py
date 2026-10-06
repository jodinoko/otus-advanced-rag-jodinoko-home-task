from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.utils import get_cache_dir

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _text(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_text(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _float(name: str, default: float) -> float:
    try:
        return float(_text(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / "data" / "knowledge_base"
    qdrant_url: str = _text("QDRANT_URL", "http://localhost:6333")
    collection_name: str = _text("QDRANT_COLLECTION", "corporate_knowledge")
    embedding_model: str = _text("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_cache_dir: Path = Path(_text("EMBEDDING_CACHE_DIR", get_cache_dir()))
    chunk_size: int = _int("CHUNK_SIZE", 700)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 100)
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 8)
    rerank_top_n: int = _int("RERANK_TOP_N", 3)
    min_rerank_score: float = float(_text("MIN_RERANK_SCORE", "0"))
    rerank_model: str = _text("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    llm_base_url: str = _text("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    llm_api_key: str = _text("LLM_API_KEY", "lm-studio")
    llm_model: str = _text("LLM_MODEL", "qwen3-14b")
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 4096)
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.0)
    llm_timeout_seconds: int = _int("LLM_TIMEOUT_SECONDS", 600)
    gradio_host: str = _text("GRADIO_HOST", "127.0.0.1")
    gradio_port: int = _int("GRADIO_PORT", 7860)


settings = Settings()
