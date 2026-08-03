## Feature: Streamlit UI for my-pysyft-rag

Description: A simple, single-page Streamlit app where users type a
question about PySyft/OpenMined docs and see the RAG's answer plus its
cited sources. Uses GroqClient (free tier) for generation — this is the
production path, matching the project's zero-cost-in-prod constraint.
Built to work both locally now (`streamlit run app.py`) and later on
Streamlit Cloud without changes.

Requirements:

- New file app.py at project root.
- On startup, cached via st.cache_resource (runs once per app instance,
  not on every user interaction):
  - Load .env
  - Try client.get_collection("pysyft_docs"); if it raises (collection
    doesn't exist yet — e.g. first run, or a fresh deploy with no
    chroma_data/), call ingest_documents() to build it from
    data/raw/*.md, then get the collection.
  - Build GroqClient(api_key=os.environ["GROQ_API_KEY"]) and
    RAGService(collection, groq_client)
- UI: a text input for the question, a submit control (button or Enter),
  and on submit: display the generated answer, then below it the list
  of source filenames used (deduplicated, since the same file can appear
  more than once in .sources).
- Show a spinner while a query is running.
- Simple page title identifying the project (e.g. "PySyft Docs
  Assistant").

Edge Cases:

- Empty question submitted — don't call RAGService (it raises
  ValueError on empty input); show an inline message asking the user to
  type a question instead.
- GROQ_API_KEY missing from environment — show a clear error message in
  the UI (not an unhandled stack trace).
- RAGService.query() raises an exception (API failure, rate limit,
  etc.) — catch it, show a friendly error message in the UI, don't crash
  the app.
- First load / cold start where ingestion is needed — show a distinct
  "setting up, first run only" spinner during that one-time step, so
  it's clear to the user why the first query is slower.

Constraints:

- Must use GroqClient, not AnthropicClient — this is the free,
  zero-cost production path.
- Reuse ingest_documents(), RAGService, and GroqClient exactly as they
  are — no changes to anything in rag/.
- Single file (app.py) — no multi-page Streamlit setup needed for this
  v1.
- Must run locally with just `streamlit run app.py`, given a .env with
  GROQ_API_KEY set.

Generate:

1. app.py — the Streamlit app as described above.
2. No automated tests for this file — UI code, manually verified by
   running the app and asking real questions. This is a deliberate scope
   decision (consistent with treating this as throwaway-v1-adjacent UI,
   not core RAG logic), not an oversight.

Run: streamlit run app.py — manually verify: ask a real question, confirm
the answer and deduplicated sources render correctly, and that the
first run's ingestion spinner appears if chroma_data/ was missing.