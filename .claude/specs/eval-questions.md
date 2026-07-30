## Feature: Candidate Q&A Set for Evals

Description: Generate a candidate set of questions, expected answers, and source documents from the PySyft docs corpus, to be curated afterward into
the final eval set.

Requirements:
- Read all .md files in data/raw/
- For each file, generate exactly 3 questions a real user might ask that
  are answered by that file's content — as if you were a human trying to understand the concepts in it
- For each question, extract the expected answer content directly from the file's text (key facts/phrases, not paraphrased or invented)
- Record which file each question/answer came from
- This is NOT a reusable script — generate the JSON directly, once, by reading the docs yourself in this session

Additionally, add these "should decline" questions by hand (NOT derived from
the docs — these test that the RAG admits it doesn't know instead of
hallucinating):

1. "What's the weather like in Buenos Aires?"
2. "Who won the 2026 World Cup?"
3. "How do I bake a chocolate cake?"
4. "What programming languages does PySyft support besides Python?"
   (plausible-sounding, adjacent to the domain, but not actually covered —
   a harder test than a totally unrelated question)
5. "What is the pricing model for using an OpenMined enclave?"
   (same idea — sounds like it could be documented, but isn't)

For all 5 of these, expected_answer should be something like:
"Should decline — not covered in the context" (not a real answer).

Output format (JSON list, one object per question):
{
  "question": "What are the core principles of Syft?",
  "expected_answer_contains": "shell-first, schema-last, transport-agnostic",
  "expected_source": "principles.md"
}

For the 5 "should decline" questions, use "expected_source": null.

Save to: evals/candidate_questions.json

Edge Cases:
- A file with little to no narrative content (e.g. mostly a changelog-style
  list or pure config) — generate fewer than 3 questions for it if 3 genuinely
  don't make sense, rather than forcing low-quality ones
- Do not generate questions that depend on information from multiple files
  combined — each question must be answerable from ONE source file

Constraints:
- No script, no reusable code, no unit tests needed for this feature — this
  is a one-time content-generation task, done directly in this session
- This is a DRAFT. After generation, review the full file manually and
  remove/fix any question that doesn't make sense, is ambiguous, or has an
  answer that isn't clearly grounded in the source text, before using it
  for evals
- Total expected size: ~72 candidate questions from docs (24 files × 3) +
  5 "should decline" questions = ~77 total

Generate:
1. evals/candidate_questions.json