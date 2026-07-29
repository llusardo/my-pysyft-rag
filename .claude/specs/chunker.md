## Feature: Document Chunker

Description: Split long markdown documents into smaller text chunks for embedding.

File: rag/chunker.py
Function: chunk_text(text: str, chunk_size: int = 500) -> list[str]

Requirements:
- Input: raw text (string) from a markdown file
- Output: list of text chunks, each ~chunk_size characters
- Break at paragraph boundaries when possible (don't cut mid-sentence if avoidable)
- Handle edge cases: empty string, text shorter than chunk_size, text with no paragraph breaks

Input/Output Example:
Input: "Paragraph one text...\n\nParagraph two text...\n\nParagraph three..."
Output: ["Paragraph one text...", "Paragraph two text...", ...]
(grouped up to ~500 chars per chunk)

Edge Cases:
- Empty string → return []
- Text shorter than chunk_size → return [text] (single chunk)
- None input → raise TypeError

Constraints:
- No external chunking libraries (write it plain Python, for learning purposes)
- Type hints on function signature
- Google-style docstring
- Write 5+ pytest tests covering: normal case, empty, short text, no paragraph breaks, exact boundary case

Generate:
1. rag/chunker.py
2. tests/test_chunker.py

Run: pytest tests/test_chunker.py -v