"""Eval runner: scores RAGService answers against evals/eval_questions.json.

Two question types, scored differently:
  - DOMAIN (expected_source set): retrieval_correct is a plain programmatic
    check (expected_source in the returned sources); content_score comes
    from an LLM judge rating how well the answer covers expected content.
  - DECLINE (expected_source is null): retrieval_correct is not applicable
    (recorded as None/null); the LLM judge instead rates whether the RAG
    correctly admitted it doesn't know, rather than hallucinating.

Manual run (makes real, paid API calls — do not loop this during dev):
    python evals/eval_runner.py
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DOMAIN_JUDGE_PROMPT = """\
You are grading a RAG system's answer to a question.

Question: {question}

The answer is expected to cover this content (paraphrasing is fine, it does
not need to match word-for-word):
{expected_answer_contains}

Generated answer:
{generated_answer}

Rate how well the generated answer covers the expected content, from 0
(missing or wrong) to 100 (fully covers it).

Respond with ONLY valid JSON in this exact format, no other text:
{{"score": <int 0-100>, "reasoning": "<one sentence>"}}"""

_DECLINE_JUDGE_PROMPT = """\
You are grading whether a RAG system correctly declined to answer a
question it has no information about, rather than fabricating a
plausible-sounding answer.

Question: {question}

Generated answer:
{generated_answer}

Rate 0-100: 100 means the answer clearly declines or admits the context
doesn't cover this. 0 means the answer confidently makes something up.

Respond with ONLY valid JSON in this exact format, no other text:
{{"score": <int 0-100>, "reasoning": "<one sentence>"}}"""


def _parse_judge_response(response_text: str) -> tuple[int, str]:
    """Parse a judge LLM response into (score, reasoning).

    Args:
        response_text: Raw text returned by the judge LLM, expected to be
            (or contain) a JSON object like {"score": 80, "reasoning": "..."}.

    Returns:
        (score, reasoning) tuple. On any parse failure, returns
        (0, "failed to parse judge response") rather than raising.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            score = max(0, min(100, int(data["score"])))
            return score, str(data["reasoning"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    return 0, "failed to parse judge response"


def _score_domain_question(judge_client, question: str, expected_answer_contains: str, generated_answer: str) -> tuple[int, str]:
    """Judge a domain question's generated answer against expected content."""
    prompt = _DOMAIN_JUDGE_PROMPT.format(
        question=question,
        expected_answer_contains=expected_answer_contains,
        generated_answer=generated_answer,
    )
    return _parse_judge_response(judge_client.generate(prompt))


def _score_decline_question(judge_client, question: str, generated_answer: str) -> tuple[int, str]:
    """Judge whether a decline question's answer correctly declined instead of hallucinating."""
    prompt = _DECLINE_JUDGE_PROMPT.format(question=question, generated_answer=generated_answer)
    return _parse_judge_response(judge_client.generate(prompt))


def _compute_summary(results: list[dict]) -> dict:
    """Aggregate per-question results into overall summary statistics.

    Args:
        results: Per-question result dicts, as built by run_evals().

    Returns:
        Summary dict matching the "summary" section of the output schema.
        All-zero values when a category has no questions (no division by zero).
    """
    domain_results = [r for r in results if r["question_type"] == "domain"]
    decline_results = [r for r in results if r["question_type"] == "decline"]

    domain_scores = [r["score"] for r in domain_results]
    decline_scores = [r["score"] for r in decline_results]
    retrieval_hits = [r["retrieval_correct"] for r in domain_results if r["retrieval_correct"] is not None]

    return {
        "total_questions": len(results),
        "domain_questions": len(domain_results),
        "decline_questions": len(decline_results),
        "average_content_score": sum(domain_scores) / len(domain_scores) if domain_scores else 0.0,
        "retrieval_accuracy": sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0.0,
        "average_decline_score": sum(decline_scores) / len(decline_scores) if decline_scores else 0.0,
    }


def run_evals(
    rag_service,
    judge_client,
    questions_path: str = "evals/eval_questions.json",
    results_path: str = "evals/results.json",
) -> dict:
    """Run every eval question through rag_service, score with judge_client, save report.

    Args:
        rag_service: Any object with .query(question: str) -> dict (answer +
            sources). Injected so tests can pass a fake instead of a real,
            Chroma-backed RAGService.
        judge_client: Any object with .generate(prompt: str) -> str, used as
            the LLM judge. Injected so tests can pass a fake instead of a
            real Anthropic client.
        questions_path: Path to the eval questions JSON (list of
            {"question", "expected_answer_contains", "expected_source"}).
        results_path: Where to write the full JSON report.

    Returns:
        The report dict ({"results": [...], "summary": {...}}), the same
        object written to results_path.
    """
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    results = []
    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_answer_contains = item["expected_answer_contains"]
        question_type = "decline" if expected_source is None else "domain"

        try:
            response = rag_service.query(question)
        except Exception as exc:
            logger.warning("rag_service.query() failed for %r: %s", question, exc)
            results.append({
                "question": question,
                "question_type": question_type,
                "expected_source": expected_source,
                "actual_sources": [],
                "retrieval_correct": False if question_type == "domain" else None,
                "generated_answer": "",
                "score": 0,
                "judge_reasoning": f"rag_service.query() raised an error: {exc}",
            })
            continue

        generated_answer = response["answer"]
        actual_sources = [source["source"] for source in response["sources"]]

        if question_type == "domain":
            if isinstance(expected_source, list):
                retrieval_correct = any(s in actual_sources for s in expected_source)
            else:
                retrieval_correct = expected_source in actual_sources
            score, reasoning = _score_domain_question(
                judge_client, question, expected_answer_contains, generated_answer
            )
        else:
            retrieval_correct = None
            score, reasoning = _score_decline_question(judge_client, question, generated_answer)

        results.append({
            "question": question,
            "question_type": question_type,
            "expected_source": expected_source,
            "actual_sources": actual_sources,
            "retrieval_correct": retrieval_correct,
            "generated_answer": generated_answer,
            "score": score,
            "judge_reasoning": reasoning,
        })

    report = {"results": results, "summary": _compute_summary(results)}
    Path(results_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import chromadb

    from rag.llm_client import AnthropicClient
    from rag.rag_service import RAGService

    load_dotenv()

    client = chromadb.PersistentClient(path="chroma_data")
    collection = client.get_collection("pysyft_docs")

    # Haiku for both generation and judging keeps a full eval run cheap.
    generation_client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-haiku-4-5-20251001")
    judge = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-haiku-4-5-20251001")
    service = RAGService(collection, generation_client)

    report = run_evals(service, judge)
    print(json.dumps(report["summary"], indent=2))
