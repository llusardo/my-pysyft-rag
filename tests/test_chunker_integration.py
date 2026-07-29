"""Integration tests: run chunker against real PySyft docs (not synthetic fixtures)."""

from pathlib import Path
import pytest

from rag.chunker import chunk_text

DATA_DIR = Path("data/raw")


def _real_doc_files():
    return sorted(DATA_DIR.glob("*.md"))


@pytest.mark.parametrize("filepath", _real_doc_files(), ids=lambda p: p.name)
def test_no_split_code_fences_in_real_docs(filepath):
    """No chunk should contain an unbalanced ``` — that would mean a code
    block got split across two chunks."""
    text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=500)

    for i, chunk in enumerate(chunks):
        assert chunk.count("```") % 2 == 0, (
            f"{filepath.name} chunk #{i} has an unbalanced code fence "
            f"(split code block)"
        )