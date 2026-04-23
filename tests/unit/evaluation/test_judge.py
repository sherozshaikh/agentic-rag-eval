from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentic_rag_eval.evaluation.judge import LLMJudge, _cohens_kappa, _extract_json
from agentic_rag_eval.schemas import EvalRecord


@dataclass
class _FakeCall:
    content: str
    model: str = "fake"
    backend: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str | None = None


class _FakeLLM:
    """Minimal stand-in for LLMClient that returns a queued content string."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.model = "fake-eval"
        self.backend = "fake"

    def complete(self, messages: Any, **_kwargs: Any) -> _FakeCall:
        for m in messages:
            content = m.content if hasattr(m, "content") else m["content"]
            self.calls.append(content)
        if not self._responses:
            return _FakeCall(content="{}")
        return _FakeCall(content=self._responses.pop(0))


class TestExtractJSON:
    def test_plain_json(self) -> None:
        assert _extract_json('{"score": 5}') == {"score": 5}

    def test_with_preamble(self) -> None:
        assert _extract_json('Here you go: {"score": 4, "rationale": "ok"}') == {
            "score": 4,
            "rationale": "ok",
        }

    def test_fenced(self) -> None:
        assert _extract_json('```json\n{"score": 3}\n```') == {"score": 3}

    def test_malformed_returns_none(self) -> None:
        assert _extract_json("not json at all") is None


class TestLLMJudge:
    def test_score_coherence_parses_score(self) -> None:
        fake = _FakeLLM(['{"score": 5, "rationale": "Flawless chain."}'])
        judge = LLMJudge(eval_llm=fake)
        out = judge.score_coherence(
            question="Who wrote Hamlet?",
            answer="Shakespeare",
            reasoning=[{"thought": "t", "action": "a", "observation": "o"}],
            evidence=[],
        )
        assert out["score"] == 5
        assert "Flawless" in out["rationale"]
        assert len(fake.calls) == 1

    def test_score_completeness_parses_score(self) -> None:
        fake = _FakeLLM(['{"score": 4, "rationale": "Minor detail missing."}'])
        judge = LLMJudge(eval_llm=fake)
        out = judge.score_completeness("Q?", "gold", "pred")
        assert out["score"] == 4

    def test_out_of_range_score_becomes_none(self) -> None:
        fake = _FakeLLM(['{"score": 7, "rationale": "oops"}'])
        judge = LLMJudge(eval_llm=fake)
        out = judge.score_completeness("Q?", "g", "p")
        assert out["score"] is None

    def test_unparseable_response(self) -> None:
        fake = _FakeLLM(["this is definitely not json"])
        judge = LLMJudge(eval_llm=fake)
        out = judge.score_completeness("Q?", "g", "p")
        assert out["score"] is None
        assert out["rationale"] == "parse_error"

    def test_llm_error_is_caught(self) -> None:
        class _Boom:
            model = "boom"
            backend = "boom"

            def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("network dead")

        judge = LLMJudge(eval_llm=_Boom())
        out = judge.score_completeness("Q?", "g", "p")
        assert out["score"] is None
        assert "llm_error" in out["rationale"]


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        pairs = [(1, 1), (0, 0), (1, 1), (0, 0)]
        assert _cohens_kappa(pairs) == pytest.approx(1.0)

    def test_no_agreement(self) -> None:
        pairs = [(1, 0), (0, 1), (1, 0), (0, 1)]
        assert _cohens_kappa(pairs) == pytest.approx(-1.0)

    def test_constant_rater(self) -> None:
        pairs = [(1, 1), (1, 0), (1, 1), (1, 0)]
        assert _cohens_kappa(pairs) == 0.0


class TestCalibrate:
    def _make_record(self, *, judge: float | None, f1: float) -> EvalRecord:
        return EvalRecord(
            eval_run_id="r1",
            question_id="q",
            question="?",
            gold_answer="g",
            predicted_answer="p",
            f1=f1,
            judge_completeness=judge,
        )

    def test_empty(self) -> None:
        judge = LLMJudge(eval_llm=_FakeLLM([]))
        out = judge.calibrate([])
        assert out["kappa"] is None
        assert out["n"] == 0

    def test_aligned(self) -> None:
        judge = LLMJudge(eval_llm=_FakeLLM([]))
        records = [
            self._make_record(judge=5, f1=0.9),
            self._make_record(judge=5, f1=0.8),
            self._make_record(judge=2, f1=0.1),
            self._make_record(judge=1, f1=0.0),
        ]
        out = judge.calibrate(records)
        assert out["n"] == 4
        assert out["kappa"] == pytest.approx(1.0)

    def test_skips_records_without_judge(self) -> None:
        judge = LLMJudge(eval_llm=_FakeLLM([]))
        records = [
            self._make_record(judge=None, f1=0.9),
            self._make_record(judge=5, f1=0.9),
        ]
        out = judge.calibrate(records)
        assert out["n"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
