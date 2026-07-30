"""Tests for rag.ingest.ingest_documents."""

import chromadb
import pytest

from rag.ingest import ingest_documents

COLLECTION_NAME = "test_pysyft_docs"


def _write_docs(dir_path, docs: dict[str, str]) -> None:
    for filename, content in docs.items():
        (dir_path / filename).write_text(content, encoding="utf-8")


def test_correct_number_of_chunks_indexed(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    # Short docs -> chunk_text keeps each as a single chunk (one per file).
    _write_docs(data_dir, {
        "a.md": "Short doc A.",
        "b.md": "Short doc B.",
    })

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    assert count == 2


def test_metadata_source_filename_attached_correctly(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    _write_docs(data_dir, {"doc_one.md": "Content of doc one."})

    ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(ids=["doc_one.md-0"])

    assert result["metadatas"][0]["source"] == "doc_one.md"
    assert result["metadatas"][0]["chunk_index"] == 0


def test_rerunning_ingest_does_not_duplicate_entries(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    _write_docs(data_dir, {"a.md": "Doc A.", "b.md": "Doc B."})

    first_count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)
    second_count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    assert first_count == second_count == 2
    assert collection.count() == 2


def test_empty_data_dir_returns_zero(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    assert count == 0


def test_non_md_files_are_ignored(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    _write_docs(data_dir, {"a.md": "Real doc."})
    (data_dir / "notes.txt").write_text("Not markdown.", encoding="utf-8")
    (data_dir / "image.png").write_bytes(b"\x89PNG\r\n")

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    assert count == 1


def test_multiple_chunks_per_file_get_sequential_ids(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    # Enough paragraphs to force chunk_text() to split within the file.
    paragraphs = "\n\n".join(f"Paragraph {i} " * 20 for i in range(10))
    _write_docs(data_dir, {"big.md": paragraphs})

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    assert count > 1
    for i in range(count):
        result = collection.get(ids=[f"big.md-{i}"])
        assert result["ids"] == [f"big.md-{i}"]


def test_three_fake_files_all_indexed_with_correct_sources(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    _write_docs(data_dir, {
        "x.md": "Doc X content.",
        "y.md": "Doc Y content.",
        "z.md": "Doc Z content.",
    })

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)
    all_docs = collection.get()

    assert count == 3
    sources = {m["source"] for m in all_docs["metadatas"]}
    assert sources == {"x.md", "y.md", "z.md"}
