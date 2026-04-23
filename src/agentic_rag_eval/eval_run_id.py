from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.prompts import get_prompt_registry


def git_sha() -> str | None:
    """Return the short git SHA of the current HEAD, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def compute_eval_run_id(
    settings: Settings | None = None,
    *,
    pipeline: str,
    dataset_split: str,
    extra: dict[str, str] | None = None,
) -> str:
    """Return a deterministic 16-char ID identifying an evaluation run."""
    settings = settings or get_settings()
    registry = get_prompt_registry()

    parts = [
        f"cfg={settings.config_hash()}",
        f"prompts={registry.version_hash()}",
        f"pipeline={pipeline}",
        f"split={dataset_split}",
        f"subset={settings.hotpotqa_subset_size}",
        f"agent_model={settings.llm_model}",
        f"eval_model={settings.eval_llm_model}",
    ]
    if extra:
        for k in sorted(extra):
            parts.append(f"{k}={extra[k]}")
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
