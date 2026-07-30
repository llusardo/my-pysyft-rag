## Feature: Document Ingestion Pipeline

Description: Process all .md files in data/raw/, chunk them, and store the
chunks with embeddings in a persistent ChromaDB collection.

File: rag/ingest.py
Function: ingest_documents(data_dir: str = "data/raw", persist_dir: str = "chroma_data", collection_name: str = "pysyft_docs") -> int

Requirements:
- Read all .md files in data_dir
- For each file, chunk it using chunk_text() from rag.chunker (reuse, don't duplicate)
- Store each chunk in a ChromaDB collection with:
  - document = chunk text
  - id = unique per chunk, e.g. f"{filename}-{chunk_index}"
  - metadata = {"source": filename, "chunk_index": chunk_index}
- Use ChromaDB's default embedding function (no external embeddings library)
- Persist to disk with PersistentClient(path=persist_dir) — no server process
- If the collection already exists, clear it and re-add (so re-running is idempotent)
- Return the total number of chunks indexed

Edge Cases:
- Empty data_dir → return 0, no error
- Non-.md files in data_dir → ignore them

Constraints:
- Reuse chunk_text() from rag/chunker.py — do not reimplement chunking here
- Type hints + Google docstring
- Write 5+ tests using a temp directory with 2-3 fake .md files:
  - Correct number of chunks indexed
  - Metadata (source filename) attached correctly
  - Re-running ingest_documents() twice doesn't duplicate entries
  - Empty directory returns 0

Generate:
1. rag/ingest.py
2. tests/test_ingest.py

Run: pytest tests/test_ingest.py -v