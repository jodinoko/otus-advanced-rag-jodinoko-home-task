from corporate_rag.citations import citation, context_block
from corporate_rag.models import RetrievedChunk


def test_pdf_citation_has_page() -> None:
    assert citation({"source": "policy.pdf", "page": 4}) == "[policy.pdf, стр. 4]"


def test_wiki_citation_has_no_fabricated_page() -> None:
    assert citation({"source": "wiki.html"}) == "[wiki.html]"


def test_context_contains_source() -> None:
    context = context_block([RetrievedChunk("text", {"source": "policy.pdf", "page": 1}, 1.0)])
    assert "[policy.pdf, стр. 1]" in context
