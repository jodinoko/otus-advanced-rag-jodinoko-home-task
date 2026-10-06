from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from corporate_rag.config import settings
from corporate_rag.models import Chunk
from corporate_rag.parsers import parse_html, parse_pdf, parse_text


def load_source_chunks(data_dir: Path = settings.data_dir) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            chunks.extend(parse_pdf(path))
        elif path.suffix.lower() in {".html", ".htm"}:
            chunks.extend(parse_html(path))
        elif path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(parse_text(path))
    if not chunks:
        raise RuntimeError(f"No supported documents found in {data_dir}")
    return chunks


def build_nodes(chunks: list[Chunk]):
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    documents = [Document(text=chunk.text, metadata=chunk.metadata) for chunk in chunks]
    return splitter.get_nodes_from_documents(documents)
