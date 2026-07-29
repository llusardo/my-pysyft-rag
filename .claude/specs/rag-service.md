## Feature: RAG Service (retrieval + generation)

Description: Core RAG orchestrator — takes a question, retrieves relevant
chunks from ChromaDB, and generates an answer using an LLM.

Files:
- rag/llm_client.py — provider-agnostic LLM adapter
- rag/rag_service.py — RAGService class

### llm_client.py

class LLMClient (base, defines the interface via a method signature):
  def generate(self, prompt: str) -> str: ...

class AnthropicClient (concrete, wraps the real Anthropic SDK):
  __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001")
  generate(self, prompt: str) -> str — calls Anthropic's messages.create(),
    returns the text content of the response

Constraint: RAGService must depend only on the LLMClient interface
(duck typing is fine — any object with .generate(prompt) -> str works),
never import anthropic directly inside rag_service.py.

### rag_service.py

class RAGService:
  __init__(self, collection, llm_client, n_results: int = 3)
    - collection: a ChromaDB collection object (already populated by ingest.py)
    - llm_client: any object implementing .generate(prompt) -> str
    - n_results: how many chunks to retrieve per query

  query(self, question: str) -> dict
    Returns: {"answer": str, "sources": list[dict]}
    Each source: {"text": str, "source": str, "chunk_index": int}

  _retrieve(self, question: str) -> list[dict]
    Queries the collection for the top n_results most similar chunks.
    Returns list of dicts with text + metadata (source, chunk_index).

  _generate(self, question: str, chunks: list[dict]) -> str
    Builds a prompt combining the question + retrieved chunk texts as context.
    Calls self.llm_client.generate(prompt).
    Returns the answer text.

Prompt template for _generate (keep it simple, plain text):
"""
Answer the question using ONLY the context below. If the context doesn't
contain the answer, say so — do not make things up.

Context:
{chunk texts, separated by ---}

Question: {question}

Answer:
"""

Edge Cases:
- No chunks retrieved (empty collection or no match) → still call the LLM
  with empty context, or return a canned "no relevant docs found" message
  (pick one, document the choice)
- question is empty string → raise ValueError

Constraints:
- RAGService must NOT import anthropic directly — only rag/llm_client.py may
- Tests for RAGService must use a FAKE llm_client (a simple test double that
  returns a canned string), never call the real Anthropic API — no cost,
  no network, deterministic
- Type hints + Google docstrings
- Write 5+ tests: normal query returns answer+sources, empty question raises,
  no matching chunks handled gracefully, sources include correct metadata,
  _generate builds prompt correctly (check via fake client capturing the prompt)

Generate:
1. rag/llm_client.py
2. rag/rag_service.py
3. tests/test_rag_service.py

Run: pytest tests/test_rag_service.py -v