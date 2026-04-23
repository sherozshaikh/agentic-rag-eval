from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import Response

from agentic_rag_eval.config import Settings
from agentic_rag_eval.serving.routes import health


def _run(coro):
    return asyncio.run(coro)


class TestDiskCheck:
    def test_disk_ok(self, tmp_path: Path) -> None:
        settings = Settings(trace_db_path=tmp_path / "t.duckdb")
        ok, msg = health._check_disk(settings)
        assert ok is True
        assert "free=" in msg


class TestDuckdbCheck:
    def test_duckdb_ok(self, tmp_path: Path) -> None:
        settings = Settings(trace_db_path=tmp_path / "t.duckdb")
        ok, msg = _run(health._check_duckdb(settings))
        assert ok is True
        assert msg == "ok"


class TestQdrantCheck:
    def test_qdrant_failure_captured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(
            trace_db_path=tmp_path / "t.duckdb",
            qdrant_host="127.0.0.1",
            qdrant_port=1,
        )
        ok, msg = _run(health._check_qdrant(settings))
        assert ok is False
        assert isinstance(msg, str)


class TestLLMCheck:
    def test_llm_check_uses_build_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(trace_db_path=tmp_path / "t.duckdb")
        stub_client = MagicMock()
        stub_client.ping = MagicMock()

        def _fake_build(_settings):
            return stub_client

        fake_module = MagicMock()
        fake_module.build_llm_client = _fake_build
        monkeypatch.setitem(__import__("sys").modules, "agentic_rag_eval.llm", fake_module)

        ok, msg = _run(health._check_llm(settings))
        assert ok is True
        stub_client.ping.assert_called_once()
        assert "ok" in msg


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_all_ok_returns_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(trace_db_path=tmp_path / "t.duckdb")

        async def _qdrant_ok(_s):
            return True, "ok"

        async def _llm_ok(_s):
            return True, "ok"

        async def _duckdb_ok(_s):
            return True, "ok"

        monkeypatch.setattr(health, "_check_qdrant", _qdrant_ok)
        monkeypatch.setattr(health, "_check_llm", _llm_ok)
        monkeypatch.setattr(health, "_check_duckdb", _duckdb_ok)
        monkeypatch.setattr(health, "_check_disk", lambda _s: (True, "ok"))

        response = Response()
        payload = await health.health_endpoint(response, settings=settings)
        assert payload.status == "ok"
        assert payload.components.qdrant
        assert response.status_code != 503

    @pytest.mark.asyncio
    async def test_any_failure_returns_degraded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(trace_db_path=tmp_path / "t.duckdb")

        async def _qdrant_fail(_s):
            return False, "connection refused"

        async def _ok(_s):
            return True, "ok"

        monkeypatch.setattr(health, "_check_qdrant", _qdrant_fail)
        monkeypatch.setattr(health, "_check_llm", _ok)
        monkeypatch.setattr(health, "_check_duckdb", _ok)
        monkeypatch.setattr(health, "_check_disk", lambda _s: (True, "ok"))

        response = Response()
        payload = await health.health_endpoint(response, settings=settings)
        assert payload.status == "degraded"
        assert payload.components.qdrant is False
        assert response.status_code == 503
