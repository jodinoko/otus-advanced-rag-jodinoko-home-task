from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from corporate_rag.models import Chunk
from corporate_rag.parsers.metadata import infer_business_metadata

HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[А-ЯA-Z][^.!?]{2,100}$")


def _header(lines: list[str], current: str) -> str:
    for line in lines:
        normalized = " ".join(line.split())
        if HEADING.match(normalized):
            return normalized
    return current


def parse_pdf(path: Path) -> list[Chunk]:
    result: list[Chunk] = []
    current_header = path.stem
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            current_header = _header(lines, current_header)
            result.append(
                Chunk(
                    text=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "header": current_header,
                        "format": "pdf",
                        **infer_business_metadata(text, path.name),
                    },
                )
            )
    return result
