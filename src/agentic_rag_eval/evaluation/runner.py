from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import duckdb

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.eval_run_id import compute_eval_run_id, git_sha
from agentic_rag_eval.evaluation.deepeval_evaluator import DeepEvalEvaluator
from agentic_rag_eval.evaluation.failure_classifier import FailureClassifier
from agentic_rag_eval.evaluation.hotpotqa_metrics import (
    exact_match,
    f1_score,
    supporting_fact_f1,
)
from agentic_rag_eval.evaluation.judge import LLMJudge
from agentic_rag_eval.evaluation.ragas_evaluator import RAGASEvaluator
from agentic_rag_eval.evaluation.retrieval_metrics import (
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import (
    EvalRecord,
    EvalRunMetadata,
    FailureMode,
    QueryResponse,
    RetrievalStrategy,
)
from agentic_rag_eval.tracing import get_trace_logger

logger = get_logger(__name__)


class Answerer(Protocol):
    """Object exposing ``answer(question) -> QueryResponse``."""

    def answer(self, question: str) -> QueryResponse: ...


AnswerFn = Callable[[str], QueryResponse]


@dataclass
class _RunCounters:
    total: int = 0
    done: int = 0
    errored: int = 0
    skipped: int = 0
    failures_classified: int = 0
    timings: list[float] = field(default_factory=list)


class EvalRunner:
    """Run evaluation for a single pipeline over a dataset with per-question checkpointing."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pipeline: str,
        dataset_split: str,
        answerer: Answerer | AnswerFn,
        enable_ragas: bool = True,
        enable_deepeval: bool = True,
        enable_judge: bool = True,
        enable_failure_classifier: bool = True,
        progress_every: int = 25,
        per_question_heavy_evals: bool = False,
        num_workers: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pipeline = pipeline
        self._dataset_split = dataset_split
        self._answerer = answerer
        self._progress_every = progress_every
        self._per_question_heavy = per_question_heavy_evals
        self._num_workers = num_workers if num_workers is not None else self._settings.eval_num_workers

        self._enable_ragas = enable_ragas
        self._enable_deepeval = enable_deepeval
        self._enable_judge = enable_judge
        self._enable_failure_classifier = enable_failure_classifier

        self._trace_logger = get_trace_logger(self._settings)
        self._write_lock = threading.Lock()

        self._ragas: RAGASEvaluator | None = None
        self._deepeval: DeepEvalEvaluator | None = None
        self._judge: LLMJudge | None = None
        self._failure: FailureClassifier | None = None

    def run(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        subset: bool = True,
    ) -> EvalRunMetadata:
        """Run the evaluation and return the finalized run metadata."""
        eval_run_id = compute_eval_run_id(
            self._settings,
            pipeline=self._pipeline,
            dataset_split=self._dataset_split,
            extra={"subset": "1" if subset else "0"},
        )

        items = list(dataset)
        metadata = EvalRunMetadata(
            eval_run_id=eval_run_id,
            git_sha=git_sha(),
            started_at=datetime.now(UTC),
            finished_at=None,
            pipeline=self._pipeline,
            dataset_split=self._dataset_split,
            num_questions=len(items),
            config_snapshot=self._settings.snapshot(),
        )
        self._trace_logger.record_eval_run(metadata)
        logger.info(
            "eval_run_started",
            extra={
                "eval_run_id": eval_run_id,
                "pipeline": self._pipeline,
                "num_questions": len(items),
                "subset": subset,
            },
        )

        already_done = self._load_done_question_ids(eval_run_id)
        if already_done:
            logger.info(
                "eval_run_resume",
                extra={"eval_run_id": eval_run_id, "done": len(already_done)},
            )

        counters = _RunCounters(total=len(items))
        completed_records: list[EvalRecord] = []

        # Build work list upfront — already_done check happens before submission,
        # so each question_id goes to exactly one worker (no duplicates possible).
        pending: list[tuple[int, str, dict[str, Any]]] = []
        for idx, item in enumerate(items, start=1):
            qid = str(item.get("question_id") or item.get("id") or f"q{idx}")
            if qid in already_done:
                counters.skipped += 1
            else:
                pending.append((idx, qid, item))

        logger.info(
            "eval_run_workers",
            extra={"num_workers": self._num_workers, "pending": len(pending)},
        )

        def _process_item(task: tuple[int, str, dict[str, Any]]) -> tuple[EvalRecord, QueryResponse, str] | None:
            _, qid, item = task
            try:
                record, response = self._process_one(
                    eval_run_id=eval_run_id,
                    question_id=qid,
                    item=item,
                )
                return record, response, qid
            except Exception as e:
                logger.error(
                    "eval_question_failed",
                    extra={
                        "eval_run_id": eval_run_id,
                        "question_id": qid,
                        "error": str(e),
                        "traceback": traceback.format_exc(limit=3),
                    },
                )
                return None

        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = {pool.submit(_process_item, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    with self._write_lock:
                        counters.errored += 1
                    continue

                record, response, qid = result

                if self._per_question_heavy:
                    _, _, item = futures[future]
                    try:
                        self._run_heavy_evals_single(record, response, item)
                    except Exception as e:
                        logger.warning(
                            "eval_heavy_single_failed",
                            extra={"question_id": qid, "error": str(e)},
                        )

                if self._enable_failure_classifier and self._is_failure(record):
                    try:
                        with self._write_lock:
                            fc = self._get_failure_classifier()
                        record.failure_mode = fc.classify(record, response)
                        if record.failure_mode != FailureMode.NONE:
                            with self._write_lock:
                                counters.failures_classified += 1
                    except Exception as e:
                        logger.warning(
                            "failure_classifier_errored",
                            extra={"question_id": qid, "error": str(e)},
                        )

                with self._write_lock:
                    try:
                        self._trace_logger.record_eval_records([record])
                    except Exception as e:
                        logger.error(
                            "eval_record_persist_failed",
                            extra={"question_id": qid, "error": str(e)},
                        )
                    completed_records.append(record)
                    counters.done += 1
                    counters.timings.append(record.latency_ms)
                    if counters.done % self._progress_every == 0:
                        logger.info(
                            "eval_progress",
                            extra={
                                "eval_run_id": eval_run_id,
                                "done": counters.done,
                                "errored": counters.errored,
                                "skipped": counters.skipped,
                                "total": counters.total,
                            },
                        )

        if not self._per_question_heavy and completed_records:
            try:
                self._run_heavy_evals_batch(completed_records, items)
                self._trace_logger.record_eval_records(completed_records)
            except Exception as e:
                logger.warning("eval_heavy_batch_failed", extra={"error": str(e)})

        self._trace_logger.finalize_eval_run(eval_run_id)
        metadata.finished_at = datetime.now(UTC)
        logger.info(
            "eval_run_finished",
            extra={
                "eval_run_id": eval_run_id,
                "done": counters.done,
                "errored": counters.errored,
                "skipped": counters.skipped,
                "failures_classified": counters.failures_classified,
            },
        )
        return metadata

    def _process_one(
        self,
        *,
        eval_run_id: str,
        question_id: str,
        item: dict[str, Any],
    ) -> tuple[EvalRecord, QueryResponse]:
        question = str(item.get("question", ""))
        gold_answer = str(item.get("answer", ""))
        gold_supporting_facts = list(item.get("supporting_facts") or [])
        gold_passage_ids = list(item.get("gold_passage_ids") or [])

        start = time.perf_counter()
        response = self._invoke_answerer(question)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if response.latency_ms <= 0:
            response.latency_ms = latency_ms

        retrieved_ids = [p.passage_id for p in response.evidence]
        retrieved_titles = [p.title or "" for p in response.evidence]

        em = exact_match(response.answer, gold_answer)
        f1 = f1_score(response.answer, gold_answer)
        sf_p, sf_r, sf_f1 = supporting_fact_f1(retrieved_titles, gold_supporting_facts)

        if gold_passage_ids:
            gold_for_retrieval: list[str] = list(gold_passage_ids)
            retrieved_ids_for_metrics: list[str] = list(retrieved_ids)
        else:
            gold_for_retrieval = [
                str(_fact_title(f)) for f in gold_supporting_facts if _fact_title(f)
            ]
            retrieved_ids_for_metrics = list(retrieved_titles)

        r5 = recall_at_k(retrieved_ids_for_metrics, gold_for_retrieval, 5)
        r10 = recall_at_k(retrieved_ids_for_metrics, gold_for_retrieval, 10)
        r20 = recall_at_k(retrieved_ids_for_metrics, gold_for_retrieval, 20)
        p5 = precision_at_k(retrieved_ids_for_metrics, gold_for_retrieval, 5)
        p10 = precision_at_k(retrieved_ids_for_metrics, gold_for_retrieval, 10)
        mrr_v = mrr(retrieved_ids_for_metrics, gold_for_retrieval)
        ndcg_v = ndcg(retrieved_ids_for_metrics, gold_for_retrieval, 10)

        strategy: RetrievalStrategy | None = None
        if response.evidence:
            strategy = response.evidence[0].source_strategy

        record = EvalRecord(
            eval_run_id=eval_run_id,
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=response.answer,
            gold_supporting_facts=[_normalize_sf(f) for f in gold_supporting_facts],
            retrieved_passage_ids=list(retrieved_ids),
            exact_match=em,
            f1=f1,
            sf_precision=sf_p,
            sf_recall=sf_r,
            sf_f1=sf_f1,
            recall_at_5=r5,
            recall_at_10=r10,
            recall_at_20=r20,
            precision_at_5=p5,
            precision_at_10=p10,
            mrr=mrr_v,
            ndcg=ndcg_v,
            latency_ms=response.latency_ms,
            strategy_used=strategy,
        )
        return record, response

    def _invoke_answerer(self, question: str) -> QueryResponse:
        if hasattr(self._answerer, "answer") and callable(self._answerer.answer):
            return self._answerer.answer(question)
        if callable(self._answerer):
            return self._answerer(question)
        raise TypeError("answerer must expose .answer(question) or be callable")

    def _get_ragas(self) -> RAGASEvaluator:
        if self._ragas is None:
            self._ragas = RAGASEvaluator(self._settings)
        return self._ragas

    def _get_deepeval(self) -> DeepEvalEvaluator:
        if self._deepeval is None:
            self._deepeval = DeepEvalEvaluator(self._settings)
        return self._deepeval

    def _get_judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(self._settings)
        return self._judge

    def _get_failure_classifier(self) -> FailureClassifier:
        if self._failure is None:
            self._failure = FailureClassifier(self._settings)
        return self._failure

    def _run_heavy_evals_single(
        self,
        record: EvalRecord,
        response: QueryResponse,
        item: dict[str, Any],
    ) -> None:
        contexts = [p.text for p in response.evidence]

        if self._enable_ragas:
            try:
                df = self._get_ragas().evaluate_batch(
                    [record.question],
                    [record.predicted_answer],
                    [contexts],
                    [record.gold_answer],
                )
                row = df.iloc[0].to_dict()
                record.ragas_faithfulness = _safe_float(row.get("faithfulness"))
                record.ragas_answer_relevancy = _safe_float(row.get("answer_relevancy"))
                record.ragas_context_precision = _safe_float(row.get("context_precision"))
                record.ragas_context_recall = _safe_float(row.get("context_recall"))
            except Exception as e:
                logger.warning("ragas_single_failed", extra={"error": str(e)})

        if self._enable_deepeval:
            try:
                scores = self._get_deepeval().evaluate(
                    {
                        "question": record.question,
                        "actual_output": record.predicted_answer,
                        "expected_output": record.gold_answer,
                        "retrieval_context": contexts,
                    }
                )
                record.deepeval_g_eval = _safe_float(scores.get("g_eval"))
                record.deepeval_hallucination = _safe_float(scores.get("hallucination"))
                record.deepeval_answer_relevancy = _safe_float(scores.get("answer_relevancy"))
            except Exception as e:
                logger.warning("deepeval_single_failed", extra={"error": str(e)})

        if self._enable_judge:
            try:
                judge = self._get_judge()
                coh = judge.score_coherence(
                    record.question,
                    record.predicted_answer,
                    response.reasoning_chain,
                    response.evidence,
                )
                com = judge.score_completeness(
                    record.question,
                    record.gold_answer,
                    record.predicted_answer,
                )
                record.judge_coherence = _safe_float(coh.get("score"))
                record.judge_completeness = _safe_float(com.get("score"))
            except Exception as e:
                logger.warning("judge_single_failed", extra={"error": str(e)})

    def _run_heavy_evals_batch(
        self,
        records: list[EvalRecord],
        items: list[dict[str, Any]],
    ) -> None:
        """Batched RAGAS + per-record judge pass."""
        if self._enable_ragas:
            try:
                ragas = self._get_ragas()
                questions = [r.question for r in records]
                answers = [r.predicted_answer for r in records]
                ground_truths = [r.gold_answer for r in records]
                contexts_list: list[list[str]] = []
                by_qid = {str(it.get("question_id") or it.get("id", "")): it for it in items}
                for r in records:
                    item = by_qid.get(r.question_id, {})
                    passages = item.get("retrieved_passages") or []
                    contexts_list.append(
                        [p.get("text", "") if isinstance(p, dict) else str(p) for p in passages]
                    )
                if any(contexts_list):
                    df = ragas.evaluate_batch(questions, answers, contexts_list, ground_truths)
                    for i, r in enumerate(records):
                        row = df.iloc[i].to_dict()
                        r.ragas_faithfulness = _safe_float(row.get("faithfulness"))
                        r.ragas_answer_relevancy = _safe_float(row.get("answer_relevancy"))
                        r.ragas_context_precision = _safe_float(row.get("context_precision"))
                        r.ragas_context_recall = _safe_float(row.get("context_recall"))
            except Exception as e:
                logger.warning("ragas_batch_failed", extra={"error": str(e)})

        if self._enable_judge:
            judge = self._get_judge()
            for r in records:
                try:
                    com = judge.score_completeness(r.question, r.gold_answer, r.predicted_answer)
                    r.judge_completeness = _safe_float(com.get("score"))
                except Exception as e:
                    logger.warning("judge_batch_failed", extra={"error": str(e)})

    def _is_failure(self, record: EvalRecord) -> bool:
        return record.exact_match == 0.0 or record.f1 < 0.5

    def _load_done_question_ids(self, eval_run_id: str) -> set[str]:
        """Return question_ids already persisted for this run."""
        try:
            with duckdb.connect(str(self._settings.trace_db_path)) as conn:
                rows = conn.execute(
                    "SELECT question_id FROM eval_records WHERE eval_run_id = ?",
                    [eval_run_id],
                ).fetchall()
                return {str(r[0]) for r in rows}
        except Exception as e:
            logger.warning(
                "eval_resume_query_failed",
                extra={"eval_run_id": eval_run_id, "error": str(e)},
            )
            return set()


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _fact_title(fact: Any) -> str | None:
    if isinstance(fact, str):
        return fact
    if isinstance(fact, dict):
        t = fact.get("title")
        return str(t) if t else None
    if isinstance(fact, list | tuple) and fact:
        return str(fact[0])
    return None


def _normalize_sf(fact: Any) -> dict[str, Any]:
    """Canonicalize a supporting fact for storage."""
    if isinstance(fact, dict):
        return {"title": str(fact.get("title", "")), "sent_id": fact.get("sent_id")}
    if isinstance(fact, list | tuple):
        return {
            "title": str(fact[0]) if fact else "",
            "sent_id": fact[1] if len(fact) > 1 else None,
        }
    return {"title": str(fact), "sent_id": None}
