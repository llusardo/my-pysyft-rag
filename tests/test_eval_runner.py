"""Tests for evals.eval_runner.run_evals.

Uses fake test doubles for both the RAG service and the judge LLM client —
zero network calls, zero API cost, fully deterministic.
"""

import json

import pytest

from evals.eval_runner import _parse_judge_response, run_evals


class FakeRAGService:
    """Test double implementing .query(question) -> dict.

    responses maps question text to either a response dict (answer +
    sources) or an Exception instance to raise when queried.
    """

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def query(self, question: str) -> dict:
        self.calls.append(question)
        result = self.responses[question]
        if isinstance(result, Exception):
            raise result
        return result


class FakeJudgeClient:
    """Test double implementing .generate(prompt) -> str.

    canned_response is either a fixed string, or a callable(prompt) -> str
    for tests that need different responses per call.
    """

    def __init__(self, canned_response="{\"score\": 80, \"reasoning\": \"covers it\"}"):
        self.canned_response = canned_response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self.canned_response):
            return self.canned_response(prompt)
        return self.canned_response


def _write_questions(path, questions):
    path.write_text(json.dumps(questions), encoding="utf-8")


def test_domain_question_retrieval_correct_when_source_matches(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "What is X?", "expected_answer_contains": "X is a thing", "expected_source": "a.md"},
    ])
    rag_service = FakeRAGService({
        "What is X?": {"answer": "X is a thing.", "sources": [{"text": "t", "source": "a.md", "chunk_index": 0}]},
    })
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    result = report["results"][0]
    assert result["question_type"] == "domain"
    assert result["retrieval_correct"] is True
    assert result["actual_sources"] == ["a.md"]


def test_domain_question_retrieval_incorrect_when_source_missing(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "What is X?", "expected_answer_contains": "X is a thing", "expected_source": "a.md"},
    ])
    rag_service = FakeRAGService({
        "What is X?": {"answer": "No idea.", "sources": [{"text": "t", "source": "other.md", "chunk_index": 0}]},
    })
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    assert report["results"][0]["retrieval_correct"] is False


def test_domain_question_retrieval_correct_when_one_of_list_sources_matches(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {
            "question": "What transport layer does syft-client use?",
            "expected_answer_contains": "Google Drive",
            "expected_source": ["connections.md", "syft-enclave-security.md", "principles.md"],
        },
    ])
    rag_service = FakeRAGService({
        "What transport layer does syft-client use?": {
            "answer": "Google Drive.",
            "sources": [{"text": "t", "source": "syft-enclave-security.md", "chunk_index": 0}],
        },
    })
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    assert report["results"][0]["retrieval_correct"] is True


def test_decline_question_uses_decline_scoring_path(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "What's the weather?", "expected_answer_contains": "Should decline", "expected_source": None},
    ])
    rag_service = FakeRAGService({
        "What's the weather?": {"answer": "I don't know, not covered.", "sources": []},
    })
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    result = report["results"][0]
    assert result["question_type"] == "decline"
    assert result["retrieval_correct"] is None
    # Decline prompt must not carry expected_answer_contains content-matching
    # language -- it asks the judge to grade decline-vs-hallucinate instead.
    prompt = judge.prompts[0]
    assert "decline" in prompt.lower()
    assert "Should decline" not in prompt


def test_judge_response_parsing_valid_and_malformed_json():
    score, reasoning = _parse_judge_response('{"score": 75, "reasoning": "mostly right"}')
    assert score == 75
    assert reasoning == "mostly right"

    score, reasoning = _parse_judge_response("not json at all")
    assert score == 0
    assert reasoning == "failed to parse judge response"

    score, reasoning = _parse_judge_response('{"score": "oops", "reasoning": 5}')
    assert score == 0
    assert reasoning == "failed to parse judge response"


def test_summary_statistics_over_two_domain_one_decline(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "Domain Q1", "expected_answer_contains": "fact1", "expected_source": "a.md"},
        {"question": "Domain Q2", "expected_answer_contains": "fact2", "expected_source": "b.md"},
        {"question": "Decline Q1", "expected_answer_contains": "Should decline", "expected_source": None},
    ])
    rag_service = FakeRAGService({
        "Domain Q1": {"answer": "fact1 answer", "sources": [{"text": "t", "source": "a.md", "chunk_index": 0}]},
        "Domain Q2": {"answer": "wrong answer", "sources": [{"text": "t", "source": "wrong.md", "chunk_index": 0}]},
        "Decline Q1": {"answer": "I don't know", "sources": []},
    })

    def scripted_judge(prompt):
        if "Domain Q1" in prompt:
            return '{"score": 100, "reasoning": "perfect"}'
        if "Domain Q2" in prompt:
            return '{"score": 40, "reasoning": "partial"}'
        return '{"score": 90, "reasoning": "declined correctly"}'

    judge = FakeJudgeClient(canned_response=scripted_judge)

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))
    summary = report["summary"]

    assert summary["total_questions"] == 3
    assert summary["domain_questions"] == 2
    assert summary["decline_questions"] == 1
    assert summary["average_content_score"] == pytest.approx(70.0)  # (100 + 40) / 2
    assert summary["retrieval_accuracy"] == pytest.approx(0.5)  # 1 correct out of 2
    assert summary["average_decline_score"] == pytest.approx(90.0)


def test_rag_service_exception_handled_without_crashing(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "Broken Q", "expected_answer_contains": "fact", "expected_source": "a.md"},
        {"question": "Fine Q", "expected_answer_contains": "fact2", "expected_source": "b.md"},
    ])
    rag_service = FakeRAGService({
        "Broken Q": RuntimeError("Chroma connection lost"),
        "Fine Q": {"answer": "fact2 answer", "sources": [{"text": "t", "source": "b.md", "chunk_index": 0}]},
    })
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    broken_result, fine_result = report["results"]
    assert broken_result["score"] == 0
    assert broken_result["retrieval_correct"] is False
    assert "error" in broken_result["judge_reasoning"].lower()
    # The rest of the run continues undisturbed.
    assert fine_result["retrieval_correct"] is True
    assert fine_result["score"] == 80


def test_judge_client_exception_handled_without_crashing(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "Broken Judge Q", "expected_answer_contains": "fact", "expected_source": "a.md"},
        {"question": "Fine Q", "expected_answer_contains": "fact2", "expected_source": "b.md"},
    ])
    rag_service = FakeRAGService({
        "Broken Judge Q": {"answer": "some answer", "sources": [{"text": "t", "source": "a.md", "chunk_index": 0}]},
        "Fine Q": {"answer": "fact2 answer", "sources": [{"text": "t", "source": "b.md", "chunk_index": 0}]},
    })

    def scripted_judge(prompt):
        if "Broken Judge Q" in prompt:
            raise RuntimeError("judge API down")
        return '{"score": 80, "reasoning": "covers it"}'

    judge = FakeJudgeClient(canned_response=scripted_judge)

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    broken_result, fine_result = report["results"]
    assert broken_result["score"] == 0
    assert "error" in broken_result["judge_reasoning"].lower()
    # retrieval_correct still reflects the real retrieval -- only judging failed.
    assert broken_result["retrieval_correct"] is True
    # The rest of the run continues undisturbed.
    assert fine_result["retrieval_correct"] is True
    assert fine_result["score"] == 80


def test_output_written_to_results_path_with_correct_schema(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [
        {"question": "Q1", "expected_answer_contains": "fact", "expected_source": "a.md"},
    ])
    rag_service = FakeRAGService({
        "Q1": {"answer": "fact answer", "sources": [{"text": "t", "source": "a.md", "chunk_index": 0}]},
    })
    judge = FakeJudgeClient()

    run_evals(rag_service, judge, str(questions_path), str(results_path))

    assert results_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))

    assert set(saved.keys()) == {"results", "summary"}
    result = saved["results"][0]
    assert set(result.keys()) == {
        "question", "question_type", "expected_source", "actual_sources",
        "retrieval_correct", "generated_answer", "score", "judge_reasoning",
    }
    assert set(saved["summary"].keys()) == {
        "total_questions", "domain_questions", "decline_questions",
        "average_content_score", "retrieval_accuracy", "average_decline_score",
    }


def test_empty_questions_file_returns_empty_results_no_crash(tmp_path):
    questions_path = tmp_path / "questions.json"
    results_path = tmp_path / "results.json"
    _write_questions(questions_path, [])
    rag_service = FakeRAGService({})
    judge = FakeJudgeClient()

    report = run_evals(rag_service, judge, str(questions_path), str(results_path))

    assert report["results"] == []
    assert report["summary"]["total_questions"] == 0
    assert report["summary"]["average_content_score"] == 0.0
    assert report["summary"]["retrieval_accuracy"] == 0.0
    assert report["summary"]["average_decline_score"] == 0.0
