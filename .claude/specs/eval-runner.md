## Feature: Eval Runner

Description: Run all questions from evals/eval_questions.json through
RAGService, score each answer with an LLM-as-judge (content quality +
retrieval correctness), and produce a JSON report with per-question scores
and an overall summary.

Requirements:

- Load questions from evals/eval_questions.json
- For each question, call rag_service.query(question) to get
  {"answer": str, "sources": list[dict]}
- Two question types, scored differently:

  DOMAIN questions (expected_source is not null):
    - expected_source may be a single source-file string (e.g. "a.md") or
      a list of source-file strings (e.g. ["a.md", "b.md"]) when more than
      one file legitimately answers the question
    - retrieval_correct: programmatic check (NOT LLM) — True if
      expected_source (or, when it's a list, ANY entry in it) appears
      among the "source" values in the returned sources, False otherwise
    - content_score (0-100): LLM-as-judge rates how well generated_answer
      covers expected_answer_contains (in its own words is fine, doesn't
      need to match verbatim)

  DECLINE questions (expected_source is null):
    - retrieval_correct: not applicable, record as null
    - content_score (0-100): LLM-as-judge rates whether the RAG correctly
      declined / admitted it doesn't know, rather than fabricating a
      plausible-sounding answer. 100 = clearly declined, 0 = confidently
      made something up.

- Judge prompts (two separate templates, one per question type — see
  Constraints for exact wording expectations)
- Judge must respond in a parseable format (e.g. JSON: {"score": int,
  "reasoning": str}) — parse this out of the LLM response
- Write full results + summary to evals/results.json

Output JSON schema:
{
  "results": [
    {
      "question": "...",
      "question_type": "domain" | "decline",
      "expected_source": "file.md" | ["file.md", ...] | null,
      "actual_sources": ["file.md", ...],
      "retrieval_correct": true | false | null,
      "generated_answer": "...",
      "score": 0-100,
      "judge_reasoning": "..."
    }
  ],
  "summary": {
    "total_questions": int,
    "domain_questions": int,
    "decline_questions": int,
    "average_content_score": float,
    "retrieval_accuracy": float,
    "average_decline_score": float
  }
}

Edge Cases:
- rag_service.query() raises an exception for some question — catch it,
  record that question's result with score=0 and judge_reasoning noting
  the error, continue with the rest (don't crash the whole run)
- Judge LLM response isn't valid/parseable JSON — treat as score=0,
  judge_reasoning="failed to parse judge response", don't crash
- Empty eval_questions.json — return empty results list, summary with
  zeros, no crash

Constraints:
- Use AnthropicClient from rag/llm_client.py for the judge — do not write
  new provider-specific code. Use model="claude-haiku-4-5-20251001"
  (cheapest model) both for the judge and confirm RAGService's own
  generation also defaults to Haiku, to keep eval runs cheap
- The runner function must accept the RAGService (or any object exposing
  .query(question) -> dict) as a parameter — dependency injection, so
  tests can pass a FAKE rag service, not a real Chroma-backed one
- Similarly, accept the judge's LLM client as a parameter (fake in tests)
- This assumes chroma_data/ is already populated (ingest_documents() has
  already run) — do not re-ingest docs in this feature
- Real cost note: running this for real makes ~40 API calls (20 questions
  × 2: one RAGService generation + one judge call each). Cheap with Haiku,
  but not free — don't call it repeatedly in a loop during development
- Tests must NOT call any real API — fake both the RAGService and the
  judge's LLM client, deterministic and free to run

Generate:
1. evals/eval_runner.py — the runner script (run manually: `python evals/eval_runner.py`)
2. tests/test_eval_runner.py — tests covering:
   - A domain question: retrieval_correct computed correctly when source
     matches / doesn't match
   - A decline question: uses the decline-scoring path, not content-scoring
   - Judge response parsing (valid JSON, and malformed JSON handled gracefully)
   - Summary statistics computed correctly over a small fake set (e.g. 2
     domain + 1 decline question)
   - rag_service.query() raising an exception is handled without crashing
   - Output written to evals/results.json in the correct schema

Run: pytest tests/test_eval_runner.py -v (all tests must pass, zero real API calls)