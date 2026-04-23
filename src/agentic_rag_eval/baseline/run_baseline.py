from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from typing import Any

import click

from agentic_rag_eval.baseline.naive_rag import BaselineRAG
from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.eval_run_id import compute_eval_run_id, git_sha
from agentic_rag_eval.evaluation.hotpotqa_metrics import exact_match, f1_score
from agentic_rag_eval.llm.client import LLMClientError, build_llm_client
from agentic_rag_eval.logging_setup import configure_logging, get_logger
from agentic_rag_eval.schemas import (
    EvalRecord,
    EvalRunMetadata,
    FailureMode,
    QueryResponse,
    RetrievalStrategy,
)
from agentic_rag_eval.tracing import get_trace_logger

logger = get_logger(__name__)


def _build_retriever(settings: Settings) -> Any:
    """Construct the dense retriever used by the baseline."""
    from qdrant_client import QdrantClient

    from agentic_rag_eval.retrieval import Retriever

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_use_grpc,
        check_compatibility=False,
    )
    return Retriever(
        client=client,
        collection_name=settings.qdrant_collection,
        settings=settings,
    )


def _load_questions(settings: Settings, *, full: bool) -> list[dict[str, Any]]:
    """Return a list of HotpotQA question dicts for the requested split."""
    from agentic_rag_eval.data import HotpotQALoader, stratified_subset

    loader = HotpotQALoader()
    if full:
        logger.info("baseline_loading_full_validation")
        df = loader.load_validation()
    else:
        logger.info(
            "baseline_loading_stratified_subset",
            extra={
                "size": settings.hotpotqa_subset_size,
                "seed": settings.hotpotqa_random_seed,
            },
        )
        train = loader.load_train()
        df = stratified_subset(
            train,
            size=settings.hotpotqa_subset_size,
            seed=settings.hotpotqa_random_seed,
        ).subset

    return df.to_dict(orient="records")


def _persist_records(
    records: list[EvalRecord],
    meta: EvalRunMetadata,
    settings: Settings,
) -> None:
    """Persist eval records and run metadata via TraceLogger."""
    tracer = get_trace_logger(settings)
    tracer.record_eval_run(meta)
    tracer.record_eval_records(records)
    tracer.finalize_eval_run(meta.eval_run_id)


def _build_eval_record(
    *,
    eval_run_id: str,
    question_row: dict[str, Any],
    response: QueryResponse,
    failure_mode: FailureMode,
) -> EvalRecord:
    """Shape a per-question `EvalRecord` from a HotpotQA row and a response."""
    question_id = str(question_row.get("question_id") or question_row.get("_id") or "")
    gold_answer = str(question_row.get("answer") or "")
    supporting_facts_raw = question_row.get("supporting_facts") or {}
    gold_supporting_facts: list[dict[str, Any]]
    if isinstance(supporting_facts_raw, dict):
        titles = supporting_facts_raw.get("title") or []
        sent_ids = supporting_facts_raw.get("sent_id") or []
        gold_supporting_facts = [
            {"title": t, "sent_id": s} for t, s in zip(titles, sent_ids, strict=False)
        ]
    elif isinstance(supporting_facts_raw, list):
        gold_supporting_facts = list(supporting_facts_raw)
    else:
        gold_supporting_facts = []

    predicted = response.answer
    return EvalRecord(
        eval_run_id=eval_run_id,
        question_id=question_id,
        question=str(question_row.get("question") or ""),
        gold_answer=gold_answer,
        predicted_answer=predicted,
        gold_supporting_facts=gold_supporting_facts,
        retrieved_passage_ids=[p.passage_id for p in response.evidence],
        exact_match=exact_match(predicted, gold_answer),
        f1=f1_score(predicted, gold_answer),
        latency_ms=response.latency_ms,
        failure_mode=failure_mode,
        strategy_used=RetrievalStrategy.DENSE,
    )


def run(
    *,
    full: bool,
    top_k: int,
    settings: Settings | None = None,
) -> EvalRunMetadata:
    """Execute the baseline run and return the finalized run metadata."""
    settings = settings or get_settings()
    configure_logging(settings)

    questions = _load_questions(settings, full=full)
    if not questions:
        raise RuntimeError("no HotpotQA questions loaded — aborting")

    dataset_split = "validation" if full else "train_subset"
    eval_run_id = compute_eval_run_id(
        settings, pipeline=BaselineRAG.PIPELINE_NAME, dataset_split=dataset_split
    )
    started_at = datetime.now(UTC)

    meta = EvalRunMetadata(
        eval_run_id=eval_run_id,
        git_sha=git_sha(),
        started_at=started_at,
        finished_at=None,
        pipeline=BaselineRAG.PIPELINE_NAME,
        dataset_split=dataset_split,
        num_questions=len(questions),
        config_snapshot=settings.snapshot(),
    )

    llm = build_llm_client(settings, role="agent")
    retriever = _build_retriever(settings)
    tracer = get_trace_logger(settings)

    pipeline = BaselineRAG(llm=llm, retriever=retriever, settings=settings, tracer=tracer)

    records: list[EvalRecord] = []
    n_errors = 0
    batch_start = time.perf_counter()

    for idx, row in enumerate(questions):
        question = str(row.get("question") or "").strip()
        if not question:
            logger.warning("baseline_skip_empty_question", extra={"index": idx})
            continue

        try:
            response = pipeline.answer(question, top_k=top_k, eval_run_id=eval_run_id)
            failure_mode = FailureMode.NONE
        except LLMClientError as exc:
            logger.error(
                "baseline_llm_failure",
                extra={"index": idx, "error": str(exc)},
            )
            response = QueryResponse(answer="", confidence="low")
            failure_mode = FailureMode.REASONING_ERROR
            n_errors += 1
        except Exception as exc:
            logger.exception(
                "baseline_unexpected_failure",
                extra={"index": idx, "error": str(exc)},
            )
            response = QueryResponse(answer="", confidence="low")
            failure_mode = FailureMode.REASONING_ERROR
            n_errors += 1

        records.append(
            _build_eval_record(
                eval_run_id=eval_run_id,
                question_row=row,
                response=response,
                failure_mode=failure_mode,
            )
        )

        if (idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - batch_start
            logger.info(
                "baseline_progress",
                extra={
                    "processed": idx + 1,
                    "total": len(questions),
                    "errors": n_errors,
                    "elapsed_s": round(elapsed, 1),
                },
            )

    meta = meta.model_copy(update={"finished_at": datetime.now(UTC)})
    _persist_records(records, meta, settings)

    logger.info(
        "baseline_run_complete",
        extra={
            "eval_run_id": eval_run_id,
            "num_questions": len(records),
            "errors": n_errors,
            "elapsed_s": round(time.perf_counter() - batch_start, 1),
        },
    )
    return meta


@click.command(context_settings={"show_default": True})
@click.option(
    "--subset/--full",
    "subset",
    default=True,
    help="Run on the stratified training subset (default) or the full validation split.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1, max=100),
    default=10,
    help="Number of dense passages to retrieve per question.",
)
def main(subset: bool, top_k: int) -> None:
    """Run the naive-RAG baseline over HotpotQA and persist results."""
    try:
        meta = run(full=not subset, top_k=top_k)
    except Exception as exc:
        click.echo(f"baseline run failed: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"baseline run complete: eval_run_id={meta.eval_run_id} "
        f"split={meta.dataset_split} n={meta.num_questions}"
    )


if __name__ == "__main__":
    main()
