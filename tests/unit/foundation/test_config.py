from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag_eval.config import Settings, get_settings


def test_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings() should pick up documented defaults when no env vars are set."""

    for var in (
        "LLM_BACKEND",
        "LLM_MODEL",
        "EVAL_LLM_BACKEND",
        "EVAL_LLM_MODEL",
        "LOG_FORMAT",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)

    assert s.llm_backend == "local"
    assert s.llm_model == "qwen2.5:7b-instruct"
    assert s.eval_llm_backend == "api"
    assert s.log_format == "json"
    assert s.log_level == "INFO"
    assert s.max_agent_steps == 5
    assert s.context_budget_tokens == 16000
    assert s.generation_reserve_tokens == 2000


@pytest.mark.parametrize("field", ["llm_backend", "eval_llm_backend"])
def test_backend_validator_rejects_invalid(field: str) -> None:
    """Both backend fields must reject values outside {local, api}."""
    with pytest.raises(ValueError, match="backend must be"):
        Settings(**{field: "gcp"})


@pytest.mark.parametrize("value", ["local", "api"])
def test_backend_validator_accepts_valid(value: str) -> None:
    """Valid backend values must be accepted."""
    s = Settings(llm_backend=value, eval_llm_backend=value)
    assert s.llm_backend == value
    assert s.eval_llm_backend == value


def test_log_format_validator_rejects_invalid() -> None:
    """`log_format` must be one of {json, text}."""
    with pytest.raises(ValueError, match="log_format must be"):
        Settings(log_format="yaml")


@pytest.mark.parametrize("value", ["json", "text"])
def test_log_format_validator_accepts_valid(value: str) -> None:
    s = Settings(log_format=value)
    assert s.log_format == value


def test_config_hash_is_deterministic() -> None:
    """Two identical configs must hash to the same value."""
    s1 = Settings(llm_model="qwen2.5:7b-instruct", llm_backend="local")
    s2 = Settings(llm_model="qwen2.5:7b-instruct", llm_backend="local")
    assert s1.config_hash() == s2.config_hash()


def test_config_hash_excludes_secrets() -> None:
    """Changing any secret must not change the config hash."""
    s1 = Settings(llm_api_key="key-one", eval_llm_api_key="eval-one", api_key="svc-one")
    s2 = Settings(llm_api_key="key-two", eval_llm_api_key="eval-two", api_key="svc-two")
    assert s1.config_hash() == s2.config_hash()


def test_config_hash_changes_on_nonsecret_field() -> None:
    """Any change to a non-secret field must produce a new hash."""
    s1 = Settings(llm_model="qwen2.5:7b-instruct")
    s2 = Settings(llm_model="qwen2.5:7b")
    assert s1.config_hash() != s2.config_hash()


def test_config_hash_returns_16_hex_chars() -> None:
    s = Settings()
    h = s.config_hash()
    assert len(h) == 16
    int(h, 16)


def test_effective_context_budget() -> None:
    s = Settings(context_budget_tokens=16000, generation_reserve_tokens=2000)
    assert s.effective_context_budget == 14000


def test_cors_origin_list_splits_and_strips() -> None:
    s = Settings(cors_origins="http://a.com, http://b.com ,http://c.com")
    assert s.cors_origin_list == ["http://a.com", "http://b.com", "http://c.com"]


def test_loading_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings should populate fields from monkeypatched env vars."""
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("CONTEXT_BUDGET_TOKENS", "32000")
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "env.duckdb"))

    get_settings.cache_clear()
    s = Settings()

    assert s.llm_backend == "api"
    assert s.llm_model == "openai/gpt-4o-mini"
    assert s.llm_api_base == "https://api.example.com/v1"
    assert s.context_budget_tokens == 32000
    assert s.trace_db_path == tmp_path / "env.duckdb"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level LRU cache should return the same instance."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


def test_snapshot_excludes_secrets() -> None:
    """snapshot() output must not contain any secret values."""
    s = Settings(
        llm_api_key="super-secret-agent",
        eval_llm_api_key="super-secret-eval",
        api_key="super-secret-svc",
    )
    snap = s.snapshot()

    assert "llm_api_key" not in snap
    assert "eval_llm_api_key" not in snap
    assert "api_key" not in snap

    assert "llm_model" in snap
    assert "llm_backend" in snap


def test_snapshot_is_json_serializable() -> None:
    """The snapshot is stored as JSON — must be serializable."""
    import json

    s = Settings()
    snap = s.snapshot()
    json.dumps(snap)
