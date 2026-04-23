from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.smoke

DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL = os.environ.get("SMOKE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "30"))
API_KEY = os.environ.get("SMOKE_API_KEY", "")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
        headers["X-API-Key"] = API_KEY
    return headers


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """HTTP client pointed at the running deployment."""
    with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT, headers=_headers()) as c:
        yield c


def _skip_if_down(client: httpx.Client) -> None:
    """Skip (don't fail) if the service simply isn't running on this host."""
    try:
        client.get("/health")
    except (httpx.ConnectError, httpx.ReadError) as exc:
        pytest.skip(f"agentic-rag-eval not reachable at {BASE_URL}: {exc}")


def test_health_endpoint_returns_200(client: httpx.Client) -> None:
    """``/health`` must return HTTP 200 with a JSON body."""
    _skip_if_down(client)

    response = client.get("/health")
    assert response.status_code == 200, (
        f"Expected 200 from /health, got {response.status_code}: {response.text}"
    )

    payload = response.json()
    assert isinstance(payload, dict), f"/health payload should be an object, got {type(payload)}"

    status = payload.get("status") or payload.get("overall") or payload.get("state")
    assert status is not None, f"/health payload missing 'status' key: {payload}"
    assert str(status).lower() in {
        "ok",
        "healthy",
        "ready",
        "up",
        "pass",
        "passing",
    }, f"/health reported unhealthy status: {status}"


def test_health_components_all_ok(client: httpx.Client) -> None:
    """Every dependency reported under ``components`` must be healthy."""
    _skip_if_down(client)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    components = (
        payload.get("components") or payload.get("checks") or payload.get("dependencies") or {}
    )
    assert components, f"/health did not report any component-level checks; payload={payload}"

    unhealthy: list[tuple[str, object]] = []
    for name, info in components.items():
        if isinstance(info, dict):
            status = info.get("status") or info.get("state") or info.get("ok")
        else:
            status = info

        if isinstance(status, bool):
            ok = status
        else:
            ok = str(status).lower() in {
                "ok",
                "healthy",
                "ready",
                "up",
                "pass",
                "passing",
                "true",
            }

        if not ok:
            unhealthy.append((name, info))

    assert not unhealthy, f"Unhealthy components reported by /health: {unhealthy}"


def test_query_endpoint_answers_simple_question(client: httpx.Client) -> None:
    """``/query`` should accept a simple question and return 200 with an answer."""
    _skip_if_down(client)

    body = {
        "question": "What is the capital of France?",
        "max_steps": 3,
    }
    response = client.post("/query", json=body)

    if response.status_code in (401, 403):
        pytest.skip(
            f"/query requires authentication (HTTP {response.status_code}). "
            "Set SMOKE_API_KEY to run this test."
        )

    assert response.status_code == 200, (
        f"Expected 200 from /query, got {response.status_code}: {response.text}"
    )

    payload = response.json()
    assert isinstance(payload, dict), f"/query payload should be an object, got {type(payload)}"

    answer = payload.get("answer") or payload.get("response") or payload.get("output")
    assert answer, f"/query response missing a non-empty 'answer' field: {payload}"
    assert isinstance(answer, str), f"/query answer should be a string, got {type(answer)}"
    assert answer.strip(), "/query returned a blank answer"
