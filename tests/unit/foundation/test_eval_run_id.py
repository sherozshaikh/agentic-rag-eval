from __future__ import annotations

import subprocess
from unittest.mock import patch

from agentic_rag_eval.config import Settings
from agentic_rag_eval.eval_run_id import compute_eval_run_id, git_sha
from agentic_rag_eval.prompts import get_prompt_registry


def test_compute_eval_run_id_is_deterministic() -> None:
    """Two invocations with the same inputs must return identical IDs."""
    get_prompt_registry.cache_clear()
    settings = Settings(llm_model="qwen2.5:7b-instruct")

    id1 = compute_eval_run_id(settings, pipeline="baseline", dataset_split="validation")
    id2 = compute_eval_run_id(settings, pipeline="baseline", dataset_split="validation")
    assert id1 == id2
    assert len(id1) == 16


def test_compute_eval_run_id_changes_when_settings_change() -> None:
    """Changing any hashed setting must produce a different ID."""
    get_prompt_registry.cache_clear()
    s1 = Settings(llm_model="qwen2.5:7b-instruct")
    s2 = Settings(llm_model="openai/gpt-4o-mini")

    id1 = compute_eval_run_id(s1, pipeline="baseline", dataset_split="validation")
    id2 = compute_eval_run_id(s2, pipeline="baseline", dataset_split="validation")
    assert id1 != id2


def test_compute_eval_run_id_changes_when_pipeline_changes() -> None:
    """Different pipelines must yield different IDs."""
    get_prompt_registry.cache_clear()
    settings = Settings()
    id1 = compute_eval_run_id(settings, pipeline="baseline", dataset_split="validation")
    id2 = compute_eval_run_id(settings, pipeline="agentic_phase2", dataset_split="validation")
    assert id1 != id2


def test_compute_eval_run_id_changes_when_split_changes() -> None:
    """Different dataset splits must yield different IDs."""
    get_prompt_registry.cache_clear()
    settings = Settings()
    id1 = compute_eval_run_id(settings, pipeline="baseline", dataset_split="validation")
    id2 = compute_eval_run_id(settings, pipeline="baseline", dataset_split="train_subset")
    assert id1 != id2


def test_compute_eval_run_id_extra_affects_id() -> None:
    """The `extra` kwargs dict must be mixed into the hash."""
    get_prompt_registry.cache_clear()
    settings = Settings()
    id1 = compute_eval_run_id(
        settings, pipeline="baseline", dataset_split="validation", extra={"seed": "1"}
    )
    id2 = compute_eval_run_id(
        settings, pipeline="baseline", dataset_split="validation", extra={"seed": "2"}
    )
    assert id1 != id2


def test_git_sha_handles_missing_git() -> None:
    """If `git` binary is missing, git_sha() must return None, not raise."""
    with patch(
        "agentic_rag_eval.eval_run_id.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        assert git_sha() is None


def test_git_sha_handles_timeout() -> None:
    """A hung git process must be gracefully handled."""
    with patch(
        "agentic_rag_eval.eval_run_id.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
    ):
        assert git_sha() is None


def test_git_sha_handles_non_zero_exit() -> None:
    """If git exits non-zero (e.g., not a repo), return None."""
    fake_result = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=128,
        stdout="",
        stderr="fatal: not a git repository",
    )
    with patch("agentic_rag_eval.eval_run_id.subprocess.run", return_value=fake_result):
        assert git_sha() is None


def test_git_sha_returns_truncated_sha() -> None:
    """A happy-path git call must return the first 12 chars of the SHA."""
    fake_result = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=0,
        stdout="abcdef1234567890deadbeef\n",
        stderr="",
    )
    with patch("agentic_rag_eval.eval_run_id.subprocess.run", return_value=fake_result):
        sha = git_sha()
    assert sha == "abcdef123456"
