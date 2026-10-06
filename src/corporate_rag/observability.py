from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


def _configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip() and os.getenv("LANGFUSE_SECRET_KEY", "").strip())


@contextmanager
def trace_answer(question: str, metadata_filters: dict[str, str] | None) -> Iterator[object | None]:
    if not _configured():
        yield None
        return

    from langfuse import Langfuse

    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )
    with client.start_as_current_observation(
        name="corporate-rag-answer",
        as_type="chain",
        input={"question": question, "metadata_filters": metadata_filters or {}},
    ) as observation:
        yield observation
    client.flush()
