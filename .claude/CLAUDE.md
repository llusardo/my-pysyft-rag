# CLAUDE.md — my-pysyft-rag

## Project

Educational RAG system over OpenMined/PySyft documentation. Built to learn RAG
fundamentals from scratch (chunking, embeddings, retrieval, generation, evals),
not to reuse existing tools (e.g. OpenMined/local-rag).

Goal: understand every component well enough to explain it, and eventually
write about it / contribute learnings back to the OpenMined community.

## Stack

- Python 3.10+
- ChromaDB (embedded/local mode — no server, no external DB engine)
- Anthropic API (Claude) for generation — **used only in development**
- LLM client must be provider-agnostic (see Architecture Rules below)
- Streamlit for the demo UI

## Folder Structure

```
my-pysyft-rag/
├── .claude/
│   ├── CLAUDE.md              (this file)
│   └── specs/                 (one spec.md per feature)
├── data/
│   └── raw/                   (source .md files copied from PySyft repo)
├── rag/
│   ├── chunker.py             (splits docs into text chunks)
│   ├── embeddings.py          (embedding logic, if customized beyond Chroma default)
│   └── rag_service.py         (core orchestrator: retrieve + generate)
├── evals/                     (quality measurement, test Q&A sets)
├── tests/                     (pytest, one test file per module)
└── venv/
```

## Architecture Rules (IMPORTANT)

### LLM client must be swappable (dependency injection)

`RAGService` must NEVER hardcode a specific provider (Anthropic, Groq, etc.)
inside its logic. The LLM client is passed in as a constructor argument.

```python
class RAGService:
    def __init__(self, vector_db, llm_client):
        self.vector_db = vector_db
        self.llm = llm_client   # any client with a compatible .create()-style method

    def query(self, question: str) -> dict:
        chunks = self._retrieve(question)
        answer = self._generate(question, chunks)
        return {"answer": answer, "sources": chunks}
```

Why: development uses Claude (Anthropic API, paid, capped spending). Production
deploy will swap to a free-tier provider (Groq, Gemini free tier, etc.)
without touching RAGService internals — just pass a different client.

### ChromaDB usage

- Embedded/local mode only (`chromadb.PersistentClient(path=...)`) — no server process
- Use Chroma's default embedding function to start (free, local, no API calls)
- Only introduce a different embeddings provider if we explicitly decide to
  compare strategies

## Code Style

- Type hints on all functions
- Docstrings: Google style
- No `print()` for debugging — use `logging`
- Naming: snake_case (functions/vars), PascalCase (classes)
- Functions < 30 lines where possible
- **Educational priority:** code should be clearly commented — this project's
  purpose is to deeply understand RAG, not just to make it work. Prefer clarity
  over cleverness.

## Testing

- pytest for all tests, one test file per module (`tests/test_<module>.py`)
- Every feature needs 5+ tests: happy path, edge cases (empty, None, boundary),
  at least 1 case that would fail with a naive implementation
- Run tests before considering any feature "done": `pytest tests/ -v`

## Things to Avoid

- ❌ Hardcoding API keys — use `.env` + `python-dotenv`
- ❌ Copying existing RAG library code directly (e.g. OpenMined/local-rag) —
  build from scratch for learning; can reference their approach conceptually
- ❌ Coupling RAGService to a specific LLM provider
- ❌ External chunking libraries — write chunking logic in plain Python (learning goal)
- ❌ Skipping tests to "move faster" — this project prioritizes understanding over speed

## Things to Always Do

- ✅ Type hints + Google docstrings
- ✅ Tests before marking a feature done
- ✅ Comment *why*, not just *what* (educational codebase)
- ✅ Keep LLM client swappable
- ✅ Track token usage / cost during development (Anthropic Console spending cap)

## References

- Spec for current feature: `.claude/specs/<feature>.md`
- Workflow this project follows: spec → agent generates → pytest → review → commit
  
## Current Status

- [x] Data collected: 24 filtered .md files from PySyft repo → `data/raw/`
- [ ] chunker.py + embeddings
- [ ] retrieval + generation (full RAG loop)
- [ ] evals (test Q&A set, quality metrics)
- [ ] Streamlit UI + deploy (swap LLM client to free-tier provider)