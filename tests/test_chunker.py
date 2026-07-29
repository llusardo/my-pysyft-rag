"""Tests for rag.chunker.chunk_text."""

import pytest

from rag.chunker import chunk_text


def test_normal_case_groups_paragraphs_up_to_chunk_size():
    """Multiple short paragraphs should be grouped together, not split one-per-chunk."""
    paragraphs = [f"Paragraph {i} text." for i in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size=50)

    # Grouping should produce fewer chunks than paragraphs (naive
    # per-paragraph chunking would give 10 chunks; grouping gives fewer).
    assert len(chunks) < len(paragraphs)
    assert all(len(c) <= 50 or "\n\n" not in c for c in chunks)
    # No content lost.
    for p in paragraphs:
        assert any(p in c for c in chunks)


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_none_input_raises_type_error():
    with pytest.raises(TypeError):
        chunk_text(None)


def test_text_shorter_than_chunk_size_returns_single_chunk():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=500)
    assert chunks == [text]


def test_text_with_no_paragraph_breaks_returns_single_chunk_when_short():
    text = "One long line with no blank lines separating anything at all."
    chunks = chunk_text(text, chunk_size=500)
    assert chunks == [text]


def test_text_with_no_paragraph_breaks_and_oversized_emits_whole_block():
    # A naive char-slicing implementation would cut this mid-sentence;
    # since there's no paragraph boundary to split on, we keep it intact.
    text = "word " * 200  # ~1000 chars, no blank lines
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_exact_boundary_case():
    """A paragraph exactly at chunk_size should fit in one chunk without splitting."""
    paragraph = "a" * 500
    chunks = chunk_text(paragraph, chunk_size=500)
    assert chunks == [paragraph]


def test_paragraph_boundary_preferred_over_mid_sentence_cut():
    """Grouping must respect paragraph breaks -- content from different
    paragraphs should never be silently merged mid-sentence."""
    p1 = "First paragraph." * 20  # ~340 chars
    p2 = "Second paragraph." * 20  # ~360 chars
    text = f"{p1}\n\n{p2}"

    chunks = chunk_text(text, chunk_size=350)

    # p1 and p2 together exceed chunk_size, so they must land in separate chunks.
    assert len(chunks) == 2
    assert chunks[0] == p1
    assert chunks[1] == p2


def test_all_content_preserved_across_chunks():
    text = "\n\n".join(f"Paragraph {i}." for i in range(5))
    chunks = chunk_text(text, chunk_size=1000)
    rejoined = "\n\n".join(chunks)
    for i in range(5):
        assert f"Paragraph {i}." in rejoined


def test_code_block_with_blank_line_stays_in_one_chunk():
    # A naive \n\n split would break this code block in two -- the blank
    # line between the import and the function body is inside the fence.
    code = "```python\nimport foo\n\ndef bar():\n    return foo.baz()\n```"
    text = f"Intro paragraph.\n\n{code}\n\nOutro paragraph."

    chunks = chunk_text(text, chunk_size=500)

    matches = [c for c in chunks if code in c]
    assert len(matches) == 1
    # No chunk should have an unbalanced fence count.
    for c in chunks:
        assert c.count("```") % 2 == 0


def test_oversized_code_block_emitted_whole():
    code_lines = "\n\n".join(f"line_{i} = {i}" for i in range(50))
    code = f"```python\n{code_lines}\n```"
    assert len(code) > 500

    chunks = chunk_text(code, chunk_size=500)

    assert len(chunks) == 1
    assert chunks[0] == code
    assert chunks[0].count("```") == 2


def test_multiple_code_blocks_each_kept_intact():
    code1 = "```python\ndef a():\n\n    return 1\n```"
    code2 = "```python\ndef b():\n\n    return 2\n```"
    text = f"First.\n\n{code1}\n\nMiddle.\n\n{code2}\n\nLast."

    chunks = chunk_text(text, chunk_size=500)

    joined = "\n\n".join(chunks)
    assert code1 in joined
    assert code2 in joined
    for c in chunks:
        assert c.count("```") % 2 == 0


def test_normal_paragraph_splitting_unaffected_by_code_fence_logic():
    paragraphs = [f"Paragraph {i} with some text." for i in range(6)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size=60)

    assert len(chunks) > 1
    for p in paragraphs:
        assert any(p in c for c in chunks)
    for c in chunks:
        assert "```" not in c
