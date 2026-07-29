"""Tests for rag.rag_service.RAGService.

Uses fake test doubles for both the Chroma collection and the LLM client —
zero network calls, zero API cost, fully deterministic.
"""

import pytest

from rag.rag_service import RAGService


class FakeCollection:
    """Stands in for a ChromaDB collection with canned query results."""

    def __init__(self, documents=None, metadatas=None):
        # Mimic Chroma's nested-list-per-query shape: one inner list per
        # query_text passed to .query(). We only ever pass one question.
        self._documents = [documents if documents is not None else []]
        self._metadatas = [metadatas if metadatas is not None else []]
        self.last_query_texts = None
        self.last_n_results = None

    def query(self, query_texts, n_results):
        self.last_query_texts = query_texts
        self.last_n_results = n_results
        return {"documents": self._documents, "metadatas": self._metadatas}


class FakeLLMClient:
    """Test double implementing .generate(prompt) -> str. Records the prompt."""

    def __init__(self, canned_response: str = "canned answer"):
        self.canned_response = canned_response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.canned_response


def _make_service(documents=None, metadatas=None, canned_response="canned answer", n_results=3):
    collection = FakeCollection(documents=documents, metadatas=metadatas)
    llm_client = FakeLLMClient(canned_response=canned_response)
    service = RAGService(collection, llm_client, n_results=n_results)
    return service, collection, llm_client


def test_normal_query_returns_answer_and_sources():
    service, _, _ = _make_service(
        documents=["Syft is a privacy tool.", "PySyft enables federated learning."],
        metadatas=[
            {"source": "intro.md", "chunk_index": 0},
            {"source": "intro.md", "chunk_index": 1},
        ],
        canned_response="Syft is a privacy-preserving framework.",
    )

    result = service.query("What is PySyft?")

    assert result["answer"] == "Syft is a privacy-preserving framework."
    assert len(result["sources"]) == 2


def test_empty_question_raises_value_error():
    service, _, _ = _make_service()

    with pytest.raises(ValueError):
        service.query("")


def test_no_matching_chunks_handled_gracefully():
    service, collection, llm_client = _make_service(documents=[], metadatas=[])

    result = service.query("Anything in an empty collection?")

    # Still calls the LLM (with empty context) rather than erroring out.
    assert result["sources"] == []
    assert llm_client.last_prompt is not None
    assert result["answer"] == "canned answer"


def test_sources_include_correct_metadata():
    service, _, _ = _make_service(
        documents=["Chunk text A.", "Chunk text B."],
        metadatas=[
            {"source": "a.md", "chunk_index": 0},
            {"source": "b.md", "chunk_index": 2},
        ],
    )

    result = service.query("Some question")

    assert result["sources"][0] == {
        "text": "Chunk text A.",
        "source": "a.md",
        "chunk_index": 0,
    }
    assert result["sources"][1] == {
        "text": "Chunk text B.",
        "source": "b.md",
        "chunk_index": 2,
    }


def test_generate_builds_prompt_correctly():
    service, _, llm_client = _make_service(
        documents=["First chunk.", "Second chunk."],
        metadatas=[
            {"source": "a.md", "chunk_index": 0},
            {"source": "b.md", "chunk_index": 0},
        ],
    )

    service.query("What is the answer?")

    prompt = llm_client.last_prompt
    assert "First chunk." in prompt
    assert "Second chunk." in prompt
    assert "---" in prompt  # chunk separator
    assert "Question: What is the answer?" in prompt
    assert "Answer:" in prompt


def test_retrieve_passes_n_results_to_collection():
    service, collection, _ = _make_service(
        documents=["Chunk."],
        metadatas=[{"source": "a.md", "chunk_index": 0}],
        n_results=5,
    )

    service.query("A question")

    assert collection.last_n_results == 5
    assert collection.last_query_texts == ["A question"]
