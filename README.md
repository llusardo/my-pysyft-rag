# PySyft Docs Assistant (my-pysyft-rag)

A RAG (Retrieval-Augmented Generation) system built from scratch over
[OpenMined/PySyft](https://github.com/OpenMined/syft) documentation.
Chunking, embeddings, retrieval, generation, evals — all hand-rolled
instead of pulled from an existing RAG framework.

**Live demo:** https://my-pysyft-rag.streamlit.app/

## Why this exists

A proof of concept to put RAG fundamentals into practice in a
controlled scope, instead of reaching for existing tooling like
OpenMined/local-rag. Chunking logic, the retrieval/generation loop, an
eval suite with dual (content + retrieval) LLM-as-judge scoring — all
built from first principles rather than assembled from a framework.

## Architecture

```
Question
   │
   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Chunking   │ ──▶ │  Embeddings  │ ──▶ │  Retrieval  │
│ (chunker.py)│     │  (ChromaDB,  │     │(rag_service │
│             │     │  local, free)│     │    .py)     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  Generation  │
                                          │ (LLM client, │
                                          │provider-agno-│
                                          │    stic)     │
                                          └──────────────┘
```

- **Chunking** (`rag/chunker.py`). Splits markdown docs into ~500-char
  chunks on paragraph boundaries. Two custom passes on top of that:
  fenced code blocks are treated as atomic units (never split
  mid-block), and headings get merged forward into their following
  content so you never end up with a heading-only chunk.
- **Embeddings + storage.** [ChromaDB](https://www.trychroma.com/) in
  embedded/local mode — no server, no external embeddings API, no cost.
  Just its default embedding function.
- **Retrieval + generation** (`rag/rag_service.py`). `RAGService` pulls
  the top-N chunks for a question, then asks an LLM to answer *only*
  from that context — and to say so when the context doesn't cover the
  question, instead of guessing.
- **LLM client** (`rag/llm_client.py`). Provider-agnostic by design:
  `RAGService` only knows about a duck-typed `.generate(prompt) -> str`
  interface, so swapping providers is adding a class here, not touching
  `RAGService` itself. Two implementations right now — `AnthropicClient`
  for development, `GroqClient` in production (free tier, zero cost).
- **Evals** (`evals/eval_runner.py`). Runs 20 curated questions (15
  answerable from the corpus, 5 deliberately out of scope) through the
  RAG and scores each answer on two independent axes: did the retrieved
  chunks actually come from the right source document (checked
  programmatically, not by an LLM), and does the generated answer cover
  the expected content (checked by a judge LLM). Generation provider is
  configurable via `EVAL_PROVIDER`; the judge stays fixed to Claude
  Haiku so scores are comparable run to run.
- **UI** (`app.py`). Single-page Streamlit app, Groq-powered, backing
  the live demo.
- **Observability** (`app.py`). [Langfuse](https://langfuse.com) (free
  tier) traces real questions asked on the live demo — latency, mostly.
  Kept separate from the eval suite and decoupled from `RAGService`, so
  the core RAG modules don't have to know observability exists.

## What broke (and how it was found)

Testing against the real 24-file corpus (not synthetic examples) turned
up two actual chunker bugs:

- **Split code fences.** The original chunker split purely on blank
  lines, which broke fenced code blocks that had blank lines inside
  them across multiple chunks. Caught with an integration test checking
  balanced triple-backtick counts per chunk — 11 of 24 files were
  affected before the fix.
- **Orphaned headings.** Some chunks ended with just a heading line,
  with the actual content pushed into the next chunk. Found the hard
  way, after the RAG gave a weak answer to a real question and I traced
  it back to bad chunk boundaries. Turned out to hit 66 of 266 chunks
  (25%) once I checked at scale.

The eval suite surfaced something more interesting on its own: two
questions looked like the RAG was confidently answering from outside
its given context — high content scores despite the "wrong" source
file being retrieved, which is normally a hallucination red flag.
Checking the actual retrieved chunk text cleared it up: the content
really was present in multiple files, just legitimate overlap between
docs, not the model making things up. A content-quality check alone
wouldn't have caught that — it took the second, independent
retrieval-correctness signal to even flag it as worth a look.

## Eval results

20 questions (15 domain, 5 "should decline"), scored with an LLM judge,
comparing the two supported generation providers:

| Metric | Claude Haiku (dev) | Groq / Llama 3.3 70B (prod) |
|---|---|---|
| Retrieval accuracy | 86.7% | 86.7% |
| Avg. content score | 65.3 / 100 | 52.1 / 100 |
| Avg. decline score | 100 / 100 | 100 / 100 |

Retrieval accuracy is identical across providers, which makes sense —
it's a property of the embedding/retrieval step and doesn't care which
LLM writes the final answer. Both providers correctly decline all 5
out-of-scope questions, so no hallucination on clearly out-of-domain
stuff. The content score gap looks worse than it is: it's a
completeness difference, not a grounding one. Given the identical
retrieved context, Llama's answers via Groq just come out more terse
than Claude's — not wrong, just shorter.

The two real retrieval misses left line up with the thinnest files in
the corpus by chunk count (`API.md`, `syft-job-README.md` — 10 and 3
chunks, against a corpus median of ~13-17). Not enough chunks there to
reliably win a similarity search, whatever the question.

## Running locally

```bash
git clone https://github.com/llusardo/my-pysyft-rag
cd my-pysyft-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Create a .env file with your API keys:
#   GROQ_API_KEY=...           (required — powers the app)
#   ANTHROPIC_API_KEY=...      (only needed to run evals — used as the judge)
#   LANGFUSE_PUBLIC_KEY=...    (optional — enables tracing; app works fine without it)
#   LANGFUSE_SECRET_KEY=...    (optional)
#   LANGFUSE_BASE_URL=https://cloud.langfuse.com   (optional)

streamlit run app.py
```

Running the eval suite:
```bash
python -m evals.eval_runner                 # Claude Haiku (default)
EVAL_PROVIDER=groq python -m evals.eval_runner   # Groq/Llama
```

Running tests (fully mocked, zero API calls, zero cost):
```bash
pytest -v
```

## Stack

Python · ChromaDB (embedded) · Anthropic API (dev) · Groq API (prod,
free tier) · Streamlit · Langfuse (observability, free tier) · pytest

## Known limitations (v1, by design)

This is a deliberately unpolished first build — a proof of concept, not
a production system. Gaps I know about: retrieval misses on
thin-content files (see Eval results above), no automated regression
tracking between eval runs, no reranking step, and Chroma's default
embeddings never got benchmarked against alternatives.
