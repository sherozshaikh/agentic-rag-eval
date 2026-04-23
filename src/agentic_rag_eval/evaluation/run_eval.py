from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from agentic_rag_eval.config import get_settings
from agentic_rag_eval.evaluation.comparison import generate_comparison_matrix
from agentic_rag_eval.evaluation.runner import EvalRunner
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


def _load_dataset(subset: bool) -> list[dict[str, Any]]:
    """Load dataset for evaluation."""
    if subset:
        path = Path("data/subsets/train_subset.parquet")
        if not path.exists():
            raise RuntimeError(f"Subset file not found at {path}. Run `make seed` first.")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    from agentic_rag_eval.data.loader import HotpotQALoader

    loader = HotpotQALoader()
    df = loader.load_validation()
    return df.to_dict(orient="records")


def _build_answerer(pipeline: str, settings: Any) -> Any:
    """Instantiate the answerer for a given pipeline name."""
    from qdrant_client import QdrantClient

    from agentic_rag_eval.llm import build_llm_client
    from agentic_rag_eval.retrieval import Retriever
    from agentic_rag_eval.tracing import get_trace_logger

    try:
        qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_use_grpc,
            check_compatibility=False,
        )
        retriever = Retriever(
            client=qdrant,
            collection_name=settings.qdrant_collection,
            settings=settings,
        )
    except Exception as e:
        click.echo(f"Failed to connect to Qdrant: {e}", err=True)
        sys.exit(2)

    llm = build_llm_client(settings, role="agent")
    tracer = get_trace_logger(settings)

    if pipeline == "baseline":
        try:
            from agentic_rag_eval.baseline import BaselineRAG
        except Exception as e:
            click.echo(f"Failed to import BaselineRAG: {e}", err=True)
            sys.exit(2)
        return BaselineRAG(llm=llm, retriever=retriever, settings=settings, tracer=tracer)

    if pipeline in {"agentic", "full"}:
        try:
            from agentic_rag_eval.agent import ReActAgent
            from agentic_rag_eval.agent.memory import MemoryStore
            from agentic_rag_eval.retrieval import AdaptiveRetriever
        except Exception as e:
            click.echo(f"Failed to import ReActAgent: {e}", err=True)
            sys.exit(2)
        adaptive = AdaptiveRetriever(retriever=retriever)
        memory = MemoryStore(settings)
        return ReActAgent(
            llm=llm,
            retriever=adaptive,
            memory=memory,
            trace_logger=tracer,
            settings=settings,
        )

    click.echo(f"Unknown pipeline: {pipeline}", err=True)
    sys.exit(2)


@click.command(context_settings={"show_default": True})
@click.option(
    "--subset/--full",
    default=True,
    help="Evaluate the 5K subset or the full validation set.",
)
@click.option(
    "--pipeline",
    type=click.Choice(["baseline", "agentic", "full"], case_sensitive=False),
    default="baseline",
    help="Which pipeline to evaluate.",
)
@click.option(
    "--dataset-split",
    default=None,
    help="Override the dataset split label (default derived from --subset/--full).",
)
@click.option(
    "--per-question-heavy/--batch-heavy",
    default=False,
    help="Run RAGAS/DeepEval/Judge per question or in a final batch.",
)
@click.option("--no-ragas", is_flag=True, default=False, help="Disable RAGAS metrics.")
@click.option("--no-deepeval", is_flag=True, default=False, help="Disable DeepEval metrics.")
@click.option("--no-judge", is_flag=True, default=False, help="Disable custom LLM-as-Judge.")
@click.option(
    "--no-failure-classifier",
    is_flag=True,
    default=False,
    help="Disable the failure-mode classifier.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Cap the number of questions evaluated (useful for smoke-tests). Omit for full run.",
)
def main(
    subset: bool,
    pipeline: str,
    dataset_split: str | None,
    per_question_heavy: bool,
    no_ragas: bool,
    no_deepeval: bool,
    no_judge: bool,
    no_failure_classifier: bool,
    limit: int | None,
) -> None:
    """Run an evaluation and print a brief summary."""
    pipeline = pipeline.lower()
    settings = get_settings()

    split = dataset_split or ("train_subset" if subset else "validation")

    click.echo(f"Loading dataset (subset={subset})...")
    dataset = _load_dataset(subset)
    if limit is not None:
        dataset = dataset[:limit]
    click.echo(f"Loaded {len(dataset)} questions.")

    click.echo(f"Building answerer for pipeline={pipeline}...")
    answerer = _build_answerer(pipeline, settings)

    runner = EvalRunner(
        settings=settings,
        pipeline=pipeline,
        dataset_split=split,
        answerer=answerer,
        enable_ragas=not no_ragas,
        enable_deepeval=not no_deepeval,
        enable_judge=not no_judge,
        enable_failure_classifier=not no_failure_classifier,
        per_question_heavy_evals=per_question_heavy,
    )

    click.echo(f"Starting eval run for pipeline={pipeline}, split={split}...")
    metadata = runner.run(dataset, subset=subset)
    click.echo(f"Run finished. eval_run_id={metadata.eval_run_id}")

    try:
        df = generate_comparison_matrix([metadata.eval_run_id])
        if not df.empty:
            click.echo("\n=== Summary ===")
            click.echo(df.to_string(index=False))
    except Exception as e:
        click.echo(f"(comparison summary failed: {e})", err=True)


if __name__ == "__main__":
    main()
