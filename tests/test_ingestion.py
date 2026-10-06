from corporate_rag.config import settings
from corporate_rag.ingestion import load_source_chunks


def test_corpus_loads_all_supported_formats() -> None:
    chunks = load_source_chunks(settings.data_dir)
    sources = {chunk.metadata["source"] for chunk in chunks}

    assert "06_information_security_policy.pdf" in sources
    assert "02_remote_work_policy.md" in sources
    assert "05_contacts_and_systems.txt" in sources
    assert "access-provisioning.html" in sources
    assert {chunk.metadata["format"] for chunk in chunks} >= {"pdf", "html", "md", "txt"}


def test_transferred_security_policy_has_page_and_security_metadata() -> None:
    chunks = load_source_chunks(settings.data_dir)
    policy_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "06_information_security_policy.pdf"]

    assert len(policy_chunks) == 3
    assert all(chunk.metadata["department"] == "Security" for chunk in policy_chunks)
    assert [chunk.metadata["page"] for chunk in policy_chunks] == [1, 2, 3]
