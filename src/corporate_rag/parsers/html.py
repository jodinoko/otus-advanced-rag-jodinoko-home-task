from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from corporate_rag.models import Chunk
from corporate_rag.parsers.metadata import infer_business_metadata


def parse_html(path: Path) -> list[Chunk]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = (soup.title.string or path.stem).strip() if soup.title else path.stem
    result: list[Chunk] = []
    business_metadata = infer_business_metadata(soup.get_text(" ", strip=True), path.name)
    header = title
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            result.append(
                Chunk(
                    text="\n".join(buffer),
                    metadata={
                        "source": path.name,
                        "header": header,
                        "format": "html",
                        **business_metadata,
                    },
                )
            )
            buffer.clear()

    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name.startswith("h"):
            flush()
            header = text
        else:
            buffer.append(text)
    flush()
    return result
