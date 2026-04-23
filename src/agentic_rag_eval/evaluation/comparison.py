from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pandas as pd

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


_MEAN_METRICS = [
    "exact_match",
    "f1",
    "sf_precision",
    "sf_recall",
    "sf_f1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "precision_at_5",
    "precision_at_10",
    "mrr",
    "ndcg",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "deepeval_g_eval",
    "deepeval_hallucination",
    "deepeval_answer_relevancy",
    "judge_coherence",
    "judge_completeness",
]


def generate_comparison_matrix(
    eval_run_ids: Sequence[str],
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per eval run containing aggregate metrics and failure-mode counts."""
    if not eval_run_ids:
        return pd.DataFrame()

    settings = settings or get_settings()
    db_path = str(settings.trace_db_path)

    placeholders = ",".join("?" for _ in eval_run_ids)
    agg_cols = ", ".join(f"AVG({m}) AS {m}" for m in _MEAN_METRICS)

    metrics_sql = f"""
        SELECT
            eval_run_id,
            COUNT(*) AS num_questions,
            {agg_cols},
            AVG(latency_ms) AS latency_ms_mean,
            quantile_cont(latency_ms, 0.5) AS latency_ms_p50,
            quantile_cont(latency_ms, 0.95) AS latency_ms_p95,
            quantile_cont(latency_ms, 0.99) AS latency_ms_p99
        FROM eval_records
        WHERE eval_run_id IN ({placeholders})
        GROUP BY eval_run_id
    """

    meta_sql = f"""
        SELECT eval_run_id, pipeline, dataset_split, git_sha, started_at, finished_at, num_questions AS planned_questions
        FROM eval_runs
        WHERE eval_run_id IN ({placeholders})
    """

    failure_sql = f"""
        SELECT eval_run_id, failure_mode, COUNT(*) AS n
        FROM eval_records
        WHERE eval_run_id IN ({placeholders})
        GROUP BY eval_run_id, failure_mode
    """

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            metrics_df = conn.execute(metrics_sql, list(eval_run_ids)).fetch_df()
            meta_df = conn.execute(meta_sql, list(eval_run_ids)).fetch_df()
            failure_df = conn.execute(failure_sql, list(eval_run_ids)).fetch_df()
    except Exception as e:
        logger.error("comparison_query_failed", extra={"error": str(e)})
        return pd.DataFrame()

    if metrics_df.empty:
        return pd.DataFrame()

    df = metrics_df.merge(meta_df, on="eval_run_id", how="left")

    if not failure_df.empty:
        fm_pivot = failure_df.pivot(
            index="eval_run_id",
            columns="failure_mode",
            values="n",
        ).fillna(0)
        fm_pivot.columns = [f"failure_{c}" for c in fm_pivot.columns]
        df = df.merge(fm_pivot.reset_index(), on="eval_run_id", how="left")

    order = {rid: i for i, rid in enumerate(eval_run_ids)}
    df["_order"] = df["eval_run_id"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return df
