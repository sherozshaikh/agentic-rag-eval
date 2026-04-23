from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from agentic_rag_eval.config import Settings
from agentic_rag_eval.evaluation.runner import EvalRunner
from agentic_rag_eval.schemas import Passage, QueryResponse, RetrievalStrategy
from agentic_rag_eval.tracing.logger import TraceLogger


def _make_settings(tmp_path: Path) -> Settings:
    """Build a Settings object writing traces to ``tmp_path``.

    We copy the defaults for everything else so tests don't depend on the
    host's ``.env`` file.
    """
    return Settings(
        trace_db_path=tmp_path / "traces.duckdb",
        mem0_storage_path=tmp_path / "mem0",
        hotpotqa_subset_size=10,
    )


def _make_response(answer: str, titles: list[str]) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        evidence=[
            Passage(
                passage_id=f"pid_{i}",
                title=t,
                text=f"text about {t}",
                source_strategy=RetrievalStrategy.HYBRID,
            )
            for i, t in enumerate(titles)
        ],
        latency_ms=1.0,
    )


def _sample_dataset() -> list[dict[str, Any]]:
    return [
        {
            "question_id": "q1",
            "question": "Who wrote Hamlet?",
            "answer": "Shakespeare",
            "supporting_facts": [["Shakespeare", 0]],
        },
        {
            "question_id": "q2",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [["Paris", 0]],
        },
        {
            "question_id": "q3",
            "question": "Who painted the Mona Lisa?",
            "answer": "Leonardo da Vinci",
            "supporting_facts": [["Leonardo da Vinci", 0]],
        },
    ]


class _ScriptedAnswerer:
    """Callable answerer that returns queued responses."""

    def __init__(self, responses: list[QueryResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, question: str) -> QueryResponse:
        self.calls.append(question)
        return self._responses.pop(0)


def _make_runner(
    tmp_path: Path,
    answerer: Any,
    *,
    pipeline: str = "baseline",
) -> tuple[EvalRunner, Settings]:
    settings = _make_settings(tmp_path)

    from agentic_rag_eval.tracing import logger as tlog_mod

    tlog_mod._instance = TraceLogger(settings.trace_db_path)

    runner = EvalRunner(
        settings=settings,
        pipeline=pipeline,
        dataset_split="train_subset",
        answerer=answerer,
        enable_ragas=False,
        enable_deepeval=False,
        enable_judge=False,
        enable_failure_classifier=False,
    )
    return runner, settings


def _fetch_records(settings: Settings, eval_run_id: str) -> list[tuple[Any, ...]]:
    with duckdb.connect(str(settings.trace_db_path)) as conn:
        return conn.execute(
            "SELECT question_id, exact_match, f1 FROM eval_records "
            "WHERE eval_run_id = ? ORDER BY question_id",
            [eval_run_id],
        ).fetchall()


class TestEvalRunner:
    def test_happy_path_checkpoints_each_question(self, tmp_path: Path) -> None:
        dataset = _sample_dataset()
        responses = [
            _make_response("Shakespeare", ["Shakespeare"]),
            _make_response("Paris", ["Paris"]),
            _make_response("Leonardo da Vinci", ["Leonardo da Vinci"]),
        ]
        runner, settings = _make_runner(tmp_path, _ScriptedAnswerer(responses))

        meta = runner.run(dataset, subset=True)
        assert meta.eval_run_id
        assert meta.num_questions == 3

        rows = _fetch_records(settings, meta.eval_run_id)
        assert len(rows) == 3
        qids = [r[0] for r in rows]
        assert qids == ["q1", "q2", "q3"]

        for _qid, em, f1 in rows:
            assert em == 1.0
            assert f1 == 1.0

    def test_error_in_answerer_is_logged_and_continues(self, tmp_path: Path) -> None:
        dataset = _sample_dataset()

        def flaky(question: str) -> QueryResponse:
            if "Hamlet" in question:
                raise RuntimeError("boom")
            return _make_response(
                "Paris" if "France" in question else "Leonardo da Vinci",
                ["t"],
            )

        runner, settings = _make_runner(tmp_path, flaky)
        meta = runner.run(dataset, subset=True)

        rows = _fetch_records(settings, meta.eval_run_id)
        qids = [r[0] for r in rows]
        assert "q1" not in qids
        assert set(qids) == {"q2", "q3"}

    def test_resume_skips_completed_questions(self, tmp_path: Path) -> None:
        dataset = _sample_dataset()
        responses_first = [
            _make_response("Shakespeare", ["Shakespeare"]),
            _make_response("Paris", ["Paris"]),
            _make_response("Leonardo da Vinci", ["Leonardo da Vinci"]),
        ]
        answerer = _ScriptedAnswerer(responses_first)
        runner, settings = _make_runner(tmp_path, answerer)
        meta = runner.run(dataset, subset=True)
        first_run_id = meta.eval_run_id
        assert len(_fetch_records(settings, first_run_id)) == 3
        assert len(answerer.calls) == 3

        second_answerer = _ScriptedAnswerer([])
        runner2, _ = _make_runner(tmp_path, second_answerer)
        meta2 = runner2.run(dataset, subset=True)
        assert meta2.eval_run_id == first_run_id

        assert second_answerer.calls == []

        assert len(_fetch_records(settings, first_run_id)) == 3

    def test_partial_resume(self, tmp_path: Path) -> None:
        dataset = _sample_dataset()

        def first_answerer(question: str) -> QueryResponse:
            if "Hamlet" in question:
                return _make_response("Shakespeare", ["Shakespeare"])
            if "France" in question:
                raise RuntimeError("simulated crash")
            return _make_response("Leonardo da Vinci", ["Leonardo da Vinci"])

        runner, settings = _make_runner(tmp_path, first_answerer)
        meta = runner.run(dataset, subset=True)

        persisted = _fetch_records(settings, meta.eval_run_id)

        qids = {r[0] for r in persisted}
        assert "q1" in qids
        assert "q3" in qids

        call_log: list[str] = []

        def retry_answerer(question: str) -> QueryResponse:
            call_log.append(question)
            if "France" in question:
                return _make_response("Paris", ["Paris"])
            return _make_response("Shakespeare", ["Shakespeare"])

        runner2, _ = _make_runner(tmp_path, retry_answerer)
        meta2 = runner2.run(dataset, subset=True)
        assert meta2.eval_run_id == meta.eval_run_id

        assert len(call_log) == 1
        assert "France" in call_log[0]

        final_rows = _fetch_records(settings, meta.eval_run_id)
        assert {r[0] for r in final_rows} == {"q1", "q2", "q3"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
