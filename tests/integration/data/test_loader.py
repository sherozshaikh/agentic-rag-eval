from __future__ import annotations

import socket
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _has_network(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


requires_network = pytest.mark.skipif(
    not _has_network(), reason="no network connectivity to HuggingFace Hub"
)
requires_datasets = pytest.mark.skipif(
    not _datasets_available(), reason="`datasets` package not installed"
)


@requires_datasets
@requires_network
def test_loader_load_train_returns_nonempty_frame(tmp_path: Path) -> None:
    from agentic_rag_eval.data.loader import HotpotQALoader

    loader = HotpotQALoader(cache_dir=tmp_path / "hf_cache")
    df = loader.load_train()
    assert len(df) > 0
    for required in ("_id", "question", "answer", "type", "level", "context"):
        assert required in df.columns


@requires_datasets
@requires_network
def test_loader_load_validation_returns_nonempty_frame(tmp_path: Path) -> None:
    from agentic_rag_eval.data.loader import HotpotQALoader

    loader = HotpotQALoader(cache_dir=tmp_path / "hf_cache")
    df = loader.load_validation()
    assert len(df) > 0
    assert set(df["type"].unique()).issubset({"bridge", "comparison"})
    assert set(df["level"].unique()).issubset({"easy", "medium", "hard"})


@requires_datasets
@requires_network
def test_loader_cache_dir_is_populated(tmp_path: Path) -> None:
    from agentic_rag_eval.data.loader import HotpotQALoader

    cache_dir = tmp_path / "hf_cache"
    loader = HotpotQALoader(cache_dir=cache_dir)
    loader.load_validation()
    assert cache_dir.exists()
    assert any(cache_dir.rglob("*")), "cache dir should contain dataset files"


@requires_datasets
@requires_network
def test_loader_question_id_alias_column(tmp_path: Path) -> None:
    from agentic_rag_eval.data.loader import HotpotQALoader

    loader = HotpotQALoader(cache_dir=tmp_path / "hf_cache")
    df = loader.load_validation()
    assert "question_id" in df.columns
    assert df["question_id"].notna().all()
