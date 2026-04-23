from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.schemas import QueryResponse, QueryType
from agentic_rag_eval.serving import deps
from agentic_rag_eval.serving.app import create_app
from agentic_rag_eval.serving.deps import get_react_agent

pytestmark = pytest.mark.integration


def _build_settings(tmp_path: Path, *, api_key: str = "") -> Settings:
    return Settings(
        trace_db_path=tmp_path / "traces.duckdb",
        mem0_storage_path=tmp_path / "mem0",
        api_key=SecretStr(api_key),
        rate_limit="10000/minute",
        cors_origins="http://localhost",
    )


def _stub_agent(response: QueryResponse) -> Any:
    agent = MagicMock()
    agent.answer.return_value = response
    return agent


@pytest.fixture
def app_and_client(tmp_path: Path, sample_query_response: QueryResponse) -> tuple[Any, TestClient]:
    deps.reset_singletons()
    settings = _build_settings(tmp_path)
    app = create_app(settings)
    agent = _stub_agent(sample_query_response)
    app.dependency_overrides[get_react_agent] = lambda: agent
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app)
    try:
        yield app, client
    finally:
        client.close()
        deps.reset_singletons()


class TestQueryEndpoint:
    def test_query_happy_path(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.post(
            "/query",
            json={"question": "Who directed Inception?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["answer"] == "The answer is 42."
        assert body["query_type"] == QueryType.BRIDGE.value
        assert body["trace_id"]

    def test_query_rejects_empty(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_query_validation_too_long(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.post("/query", json={"question": "x" * 5000})
        assert resp.status_code == 422

    def test_query_agent_exception_returns_500(
        self, app_and_client: tuple[Any, TestClient]
    ) -> None:
        app, client = app_and_client
        failing_agent = MagicMock()
        failing_agent.answer.side_effect = RuntimeError("boom")
        app.dependency_overrides[get_react_agent] = lambda: failing_agent

        resp = client.post("/query", json={"question": "What is X?"})
        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body or "detail" in body


class TestAuthGate:
    def test_auth_required_when_api_key_set(
        self, tmp_path: Path, sample_query_response: QueryResponse
    ) -> None:
        deps.reset_singletons()
        settings = _build_settings(tmp_path, api_key="secret")
        app = create_app(settings)
        app.dependency_overrides[get_react_agent] = lambda: _stub_agent(sample_query_response)
        app.dependency_overrides[get_settings] = lambda: settings

        with TestClient(app) as client:
            r1 = client.post("/query", json={"question": "hi"})
            assert r1.status_code == 401

            r2 = client.post(
                "/query",
                json={"question": "hi"},
                headers={"Authorization": "Bearer secret"},
            )
            assert r2.status_code == 200
        deps.reset_singletons()


class TestMetaRoutes:
    def test_metrics_endpoint(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "all_time" in body
        assert "last_24h" in body

    def test_eval_runs_empty(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/eval-runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_dashboard_renders(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "agentic-rag-eval" in resp.text
        assert "Overview" in resp.text

    def test_root_redirects_to_dashboard(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"].endswith("/dashboard")

    def test_traces_page_renders(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/dashboard/traces")
        assert resp.status_code == 200
        assert "Traces" in resp.text


class TestBatchEvalLock:
    def test_missing_run_returns_404(self, app_and_client: tuple[Any, TestClient]) -> None:
        _, client = app_and_client
        resp = client.get("/eval-runs/does-not-exist")
        assert resp.status_code == 404
