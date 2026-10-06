from __future__ import annotations

import re
from pathlib import Path

from corporate_rag.models import Chunk
from corporate_rag.parsers.metadata import infer_business_metadata

HEADING = re.compile(r"^(?:#{1,6}\s+|\d+\.\s+|[А-ЯA-Z][А-ЯA-Z\s-]{4,})")


def parse_text(path: Path) -> list[Chunk]:
    raw_text = path.read_text(encoding="utf-8")
    business_metadata = infer_business_metadata(raw_text, path.name)
    title = path.stem.replace("_", " ")
    header = title
    buffer: list[str] = []
    result: list[Chunk] = []

    def flush() -> None:
        if buffer:
            result.append(
                Chunk(
                    text="\n".join(buffer),
                    metadata={
                        "source": path.name,
                        "header": header,
                        "format": path.suffix.lower().lstrip("."),
                        **business_metadata,
                    },
                )
            )
            buffer.clear()

    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if HEADING.match(text):
            flush()
            header = re.sub(r"^#+\s*", "", text)
        else:
            buffer.append(text)
    flush()
    return result
