"""Core RAG orchestrator: retrieve relevant chunks, then generate an answer.

RAGService never imports a specific LLM provider SDK — it only relies on
the llm_client duck-typed interface (.generate(prompt) -> str), so the
underlying provider can be swapped (dev: Anthropic, prod: free-tier) by
passing a different client into the constructor. See rag/llm_client.py.
"""

import logging

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Answer the question using ONLY the context below. If the context doesn't
contain the answer, say so — do not make things up.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """Retrieves context from a Chroma collection, then asks an LLM to answer."""

    def __init__(self, collection, llm_client, n_results: int = 3) -> None:
        """Initialize the RAG service.

        Args:
            collection: A ChromaDB collection object, already populated
                (e.g. by rag/ingest.py).
            llm_client: Any object implementing .generate(prompt: str) -> str.
            n_results: Number of chunks to retrieve per query.
        """
        self.collection = collection
        self.llm_client = llm_client
        self.n_results = n_results

    def query(self, question: str) -> dict:
        """Answer a question by retrieving context and calling the LLM.

        Args:
            question: The user's question.

        Returns:
            Dict with "answer" (str) and "sources" (list of dicts, each
            with "text", "source", "chunk_index").

        Raises:
            ValueError: If question is an empty string.
        """
        if question == "":
            raise ValueError("question must not be empty")

        chunks = self._retrieve(question)
        answer = self._generate(question, chunks)
        return {"answer": answer, "sources": chunks}

    def _retrieve(self, question: str) -> list[dict]:
        """Query the collection for the top n_results most similar chunks.

        Args:
            question: The user's question, used as the query text.

        Returns:
            List of dicts with "text", "source", "chunk_index". Empty list
            if the collection has no matches (e.g. it's empty).
        """
        results = self.collection.query(query_texts=[question], n_results=self.n_results)

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        return [
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
            }
            for doc, meta in zip(documents, metadatas)
        ]

    def _generate(self, question: str, chunks: list[dict]) -> str:
        """Build a context prompt from chunks and ask the LLM to answer.

        Edge case: if no chunks were retrieved, we still call the LLM with
        an empty context section rather than short-circuiting with a
        canned message — the prompt template already instructs the model
        to say so when context doesn't contain the answer, which keeps
        this method (and its behavior) simple and consistent for callers.

        Args:
            question: The user's question.
            chunks: Retrieved chunks, as returned by _retrieve().

        Returns:
            The LLM-generated answer text.
        """
        context = "\n---\n".join(chunk["text"] for chunk in chunks)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        return self.llm_client.generate(prompt)
