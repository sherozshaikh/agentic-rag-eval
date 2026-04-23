from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.data.index_passages import index_passages
from agentic_rag_eval.data.loader import HotpotQALoader
from agentic_rag_eval.data.passages import (
    extract_unique_passages,
    passages_to_dataframe,
)
from agentic_rag_eval.data.subset import StratifiedSubsetResult, stratified_subset
from agentic_rag_eval.data.validate import ChiSquaredReport, chi_squared_validate
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data")
SUBSETS_SUBDIR = "subsets"
PROCESSED_SUBDIR = "processed"


def _ensure_dirs(output_root: Path) -> tuple[Path, Path]:
    subsets_dir = output_root / SUBSETS_SUBDIR
    processed_dir = output_root / PROCESSED_SUBDIR
    subsets_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return subsets_dir, processed_dir


def _save_subset(subset: pd.DataFrame, path: Path) -> None:
    logger.info(
        "Saving train subset parquet",
        extra={"path": str(path), "rows": len(subset)},
    )
    subset.to_parquet(path, index=False)


def _save_passages(passages_df: pd.DataFrame, path: Path) -> None:
    logger.info(
        "Saving passages parquet",
        extra={"path": str(path), "rows": len(passages_df)},
    )
    passages_df.to_parquet(path, index=False)


def _save_report(
    subset_result: StratifiedSubsetResult,
    chi_report: ChiSquaredReport,
    path: Path,
) -> None:
    payload = {
        "stratification": subset_result.to_report(),
        "chi_squared": chi_report.as_dict(),
    }
    logger.info("Saving stratification report", extra={"path": str(path)})
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_seed_pipeline(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    settings: Settings | None = None,
    skip_index: bool = False,
    recreate_collection: bool = False,
) -> dict[str, Path]:
    """Run the full seed pipeline and return the written artifact paths."""
    settings = settings or get_settings()
    subsets_dir, processed_dir = _ensure_dirs(output_root)

    subset_parquet = subsets_dir / "train_subset.parquet"
    passages_parquet = processed_dir / "passages.parquet"
    report_json = subsets_dir / "stratification_report.json"

    loader = HotpotQALoader()
    train_df = loader.load_train()

    subset_result = stratified_subset(
        train_df,
        size=settings.hotpotqa_subset_size,
        seed=settings.hotpotqa_random_seed,
    )

    chi_report = chi_squared_validate(subset_result.subset, train_df, assert_pass=True)

    _save_subset(subset_result.subset, subset_parquet)
    _save_report(subset_result, chi_report, report_json)

    passages = extract_unique_passages(subset_result.subset)
    if not passages:
        raise RuntimeError("No passages extracted from the subset — aborting seed.")
    passages_df = passages_to_dataframe(passages)
    _save_passages(passages_df, passages_parquet)

    if not skip_index:
        try:
            index_passages(
                passages_path=passages_parquet,
                settings=settings,
                recreate=recreate_collection,
            )
        except RuntimeError as exc:
            logger.error(
                "Qdrant indexing step failed — parquet outputs still written",
                extra={"error": str(exc)},
            )
            raise
    else:
        logger.info("Skipping Qdrant indexing (skip_index=True)")

    return {
        "train_subset": subset_parquet,
        "passages": passages_parquet,
        "report": report_json,
    }


@click.command()
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_ROOT,
    show_default=True,
    help="Root directory for `data/subsets` and `data/processed`.",
)
@click.option(
    "--skip-index",
    is_flag=True,
    default=False,
    help="Write parquet artifacts but do not index into Qdrant.",
)
@click.option(
    "--recreate-collection",
    is_flag=True,
    default=False,
    help="Drop and recreate the Qdrant collection before indexing.",
)
def main(output_root: Path, skip_index: bool, recreate_collection: bool) -> None:
    """Seed HotpotQA data: subset, validate, extract passages, index Qdrant."""
    try:
        outputs = run_seed_pipeline(
            output_root=output_root,
            skip_index=skip_index,
            recreate_collection=recreate_collection,
        )
    except AssertionError as exc:
        click.echo(f"CHI-SQUARED FAILED: {exc}", err=True)
        sys.exit(2)
    except RuntimeError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    click.echo("Seed pipeline complete. Wrote:")
    for name, path in outputs.items():
        click.echo(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
