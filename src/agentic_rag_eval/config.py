from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central project configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_backend: str = Field(default="local", description="`local` or `api`")
    llm_model: str = Field(default="qwen2.5:7b-instruct")
    llm_api_base: str = Field(default="http://localhost:11434/v1")
    llm_api_key: SecretStr = Field(default=SecretStr("ollama"))
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_base_delay: float = Field(default=1.0, gt=0)
    llm_retry_max_delay: float = Field(default=8.0, gt=0)

    eval_llm_backend: str = Field(default="api")
    eval_llm_model: str = Field(default="google/gemini-2.5-flash")
    eval_llm_api_base: str = Field(default="https://openrouter.ai/api/v1")
    eval_llm_api_key: SecretStr = Field(default=SecretStr(""))
    eval_llm_timeout_seconds: float = Field(default=30.0, gt=0)

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_collection: str = Field(default="hotpotqa_passages")
    qdrant_use_grpc: bool = Field(default=False)

    dense_model: str = Field(default="BAAI/bge-small-en-v1.5")
    sparse_model: str = Field(default="Qdrant/bm25")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    max_agent_steps: int = Field(default=5, ge=1)
    context_budget_tokens: int = Field(default=16000, ge=1024)
    generation_reserve_tokens: int = Field(default=2000, ge=256)

    hotpotqa_subset_size: int = Field(default=5000, ge=1)
    hotpotqa_random_seed: int = Field(default=42)

    trace_db_path: Path = Field(default=Path("./traces/traces.duckdb"))
    trace_retention_days: int = Field(default=30, ge=1)

    mem0_storage_path: Path = Field(default=Path("./mem0_storage"))

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_key: SecretStr = Field(default=SecretStr(""))
    rate_limit: str = Field(default="60/minute")
    cors_origins: str = Field(default="http://localhost:8000")

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    @field_validator("llm_backend", "eval_llm_backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        if v not in {"local", "api"}:
            raise ValueError(f"backend must be 'local' or 'api', got: {v!r}")
        return v

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        if v not in {"json", "text"}:
            raise ValueError(f"log_format must be 'json' or 'text', got: {v!r}")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_context_budget(self) -> int:
        """Usable context after reserving generation tokens."""
        return self.context_budget_tokens - self.generation_reserve_tokens

    def config_hash(self) -> str:
        """Return a deterministic 16-char hash of non-secret config fields."""
        exclude = {"llm_api_key", "eval_llm_api_key", "api_key"}
        payload = self.model_dump(mode="json", exclude=exclude)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def snapshot(self) -> dict[str, object]:
        """Return a serializable config snapshot with secrets excluded."""
        exclude = {"llm_api_key", "eval_llm_api_key", "api_key"}
        return self.model_dump(mode="json", exclude=exclude)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
