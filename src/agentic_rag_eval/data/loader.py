from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_HF_REPO = "hotpotqa/hotpot_qa"
DEFAULT_HF_CONFIG = "distractor"

_REQUIRED_FIELDS = (
    "_id",
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
)


class HotpotQALoader:
    """Load HotpotQA splits as pandas DataFrames."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        repo: str = DEFAULT_HF_REPO,
        config: str = DEFAULT_HF_CONFIG,
        trust_remote_code: bool = True,
    ) -> None:
        self.repo = repo
        self.config = config
        self.trust_remote_code = trust_remote_code
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_train(self) -> pd.DataFrame:
        """Return the HotpotQA training split."""
        return self._load_split("train")

    def load_validation(self) -> pd.DataFrame:
        """Return the HotpotQA validation split."""
        return self._load_split("validation")

    def _load_split(self, split: str) -> pd.DataFrame:
        """Load a single split as a DataFrame."""
        logger.info(
            "Loading HotpotQA split",
            extra={"repo": self.repo, "config": self.config, "split": split},
        )

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required to load HotpotQA. "
                "Install it via `pip install datasets`."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "path": self.repo,
            "name": self.config,
            "split": split,
        }

        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)

        try:
            ds = load_dataset(**load_kwargs)
        except Exception as exc:
            logger.exception("Failed to load HotpotQA from HuggingFace")
            raise RuntimeError(
                f"Could not load {self.repo}:{self.config} split={split!r}. "
                "Check your network connection and HuggingFace cache."
            ) from exc

        rows: list[dict[str, Any]] = list(ds)
        if not rows:
            raise RuntimeError(
                f"HotpotQA split {split!r} returned zero rows — the dataset appears empty."
            )

        rows = [self._normalize_row(row) for row in rows]

        self._validate_schema(rows[0], split)

        df = pd.DataFrame(rows)

        if "question_id" not in df.columns and "_id" in df.columns:
            df["question_id"] = df["_id"].astype(str)

        logger.info(
            "Loaded HotpotQA split",
            extra={"split": split, "num_rows": len(df)},
        )
        return df

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        """Normalize external dataset row to internal schema."""
        row = dict(row)

        if "_id" not in row and "id" in row:
            row["_id"] = row["id"]

        return row

    @staticmethod
    def _validate_schema(row: dict[str, Any], split: str) -> None:
        missing = [f for f in _REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"HotpotQA split {split!r} is missing required fields: {missing}")
