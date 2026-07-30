"""Tests for rag.ingest.ingest_documents."""

import chromadb
import pytest
from chromadb.api.client import Client as ChromaClient

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


def test_add_calls_are_batched_by_max_batch_size(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_data"

    # 7 fake docs, each short enough to produce exactly 1 chunk.
    _write_docs(data_dir, {f"doc{i}.md": f"Doc {i} content." for i in range(7)})

    # Real get_max_batch_size() returns thousands+; force a small value
    # (3) so 7 documents actually exercise the batching loop.
    monkeypatch.setattr(ChromaClient, "get_max_batch_size", lambda self: 3)

    # Spy on collection.add() by wrapping the collection create_collection()
    # hands back, so we can record what each individual .add() call received
    # without changing ingest.py itself.
    add_call_sizes = []
    original_create_collection = ChromaClient.create_collection

    def spy_create_collection(self, name, **kwargs):
        collection = original_create_collection(self, name, **kwargs)
        original_add = collection.add

        def spy_add(documents, ids, metadatas):
            add_call_sizes.append(len(documents))
            return original_add(documents=documents, ids=ids, metadatas=metadatas)

        collection.add = spy_add
        return collection

    monkeypatch.setattr(ChromaClient, "create_collection", spy_create_collection)

    count = ingest_documents(str(data_dir), str(persist_dir), COLLECTION_NAME)

    assert count == 7
    assert len(add_call_sizes) > 1  # multiple batches, not one big .add()
    assert all(size <= 3 for size in add_call_sizes)  # no batch over the limit
    assert sum(add_call_sizes) == 7  # nothing lost or duplicated across batches

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)
    assert collection.count() == 7
