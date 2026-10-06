from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
