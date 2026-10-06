from corporate_rag.models import RetrievedChunk
from corporate_rag.retrieval import deduplicate_chunks


def test_deduplicate_chunks_removes_only_identical_source_fragments() -> None:
    duplicate = RetrievedChunk(
        text="Запрос подаётся за два дня.",
        metadata={"source": "policy.pdf", "page": 2, "header": "Оформление"},
        score=0.9,
    )
    repeated = RetrievedChunk(
        text=duplicate.text,
        metadata=duplicate.metadata,
        score=0.8,
    )
    another_page = RetrievedChunk(
        text="Офисные дни: вторник, среда, четверг.",
        metadata={"source": "policy.pdf", "page": 1, "header": "Режим работы"},
        score=0.7,
    )

    result = deduplicate_chunks([duplicate, repeated, another_page])

    assert result == [duplicate, another_page]
