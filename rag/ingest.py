"""Document ingestion pipeline: chunk markdown docs and store in ChromaDB."""

import logging
from pathlib import Path

import chromadb

from rag.chunker import chunk_text

logger = logging.getLogger(__name__)


def ingest_documents(
    data_dir: str = "data/raw",
    persist_dir: str = "chroma_data",
    collection_name: str = "pysyft_docs",
) -> int:
    """Chunk all markdown files in a directory and store them in ChromaDB.

    Reads every .md file in data_dir, splits it into chunks with
    chunk_text(), and upserts each chunk as a document in a persistent
    Chroma collection using Chroma's default (local, free) embedding
    function. Re-running this function is idempotent: the collection is
    cleared before re-adding, so repeated runs never duplicate entries.

    Args:
        data_dir: Directory containing source .md files.
        persist_dir: Path where ChromaDB persists its data on disk.
        collection_name: Name of the Chroma collection to (re)populate.

    Returns:
        Total number of chunks indexed across all files.
    """
    md_files = sorted(Path(data_dir).glob("*.md"))

    client = chromadb.PersistentClient(path=persist_dir)

    # Idempotency: drop any prior collection with this name before
    # re-adding, so re-running ingestion doesn't accumulate duplicates.
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    collection = client.create_collection(collection_name)

    if not md_files:
        return 0

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for filepath in md_files:
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{filepath.name}-{chunk_index}")
            metadatas.append({"source": filepath.name, "chunk_index": chunk_index})

    if documents:
        # Chroma rejects .add() calls above its max batch size, which can
        # be far smaller than the full corpus (e.g. 5461) -- split into
        # aligned slices and add sequentially so ingest doesn't fail once
        # the docs grow past that limit.
        batch_size = client.get_max_batch_size()
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.add(
                documents=documents[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )

    logger.info("Indexed %d chunks from %d files", len(documents), len(md_files))
    return len(documents)
