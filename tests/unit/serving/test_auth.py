from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from agentic_rag_eval.config import Settings
from agentic_rag_eval.serving.auth import _extract_bearer_token, verify_api_key


def _call(authorization: str | None, api_key: str) -> None:
    settings = Settings(api_key=SecretStr(api_key))
    asyncio.run(verify_api_key(authorization=authorization, settings=settings))


class TestExtractBearerToken:
    def test_none_header(self) -> None:
        assert _extract_bearer_token(None) is None

    def test_empty_header(self) -> None:
        assert _extract_bearer_token("") is None

    def test_missing_scheme(self) -> None:
        assert _extract_bearer_token("onlyvalue") is None

    def test_wrong_scheme(self) -> None:
        assert _extract_bearer_token("Basic abc") is None

    def test_case_insensitive_scheme(self) -> None:
        assert _extract_bearer_token("bearer tok") == "tok"
        assert _extract_bearer_token("Bearer tok") == "tok"
        assert _extract_bearer_token("BEARER tok") == "tok"

    def test_strips_token(self) -> None:
        assert _extract_bearer_token("Bearer   hello  ") == "hello"

    def test_empty_token_rejected(self) -> None:
        assert _extract_bearer_token("Bearer    ") is None


class TestVerifyApiKey:
    def test_auth_disabled_when_key_empty(self) -> None:
        _call(None, "")
        _call("Bearer whatever", "")

    def test_valid_key_accepted(self) -> None:
        _call("Bearer secret123", "secret123")

    def test_missing_header_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _call(None, "secret123")
        assert exc.value.status_code == 401

    def test_wrong_key_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _call("Bearer nope", "secret123")
        assert exc.value.status_code == 401

    def test_malformed_header_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _call("Token abc", "secret123")
        assert exc.value.status_code == 401

    def test_www_authenticate_header_set(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _call(None, "secret123")
        assert exc.value.headers is not None
        assert exc.value.headers.get("WWW-Authenticate") == "Bearer"
