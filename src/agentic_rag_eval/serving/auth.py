from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Extract a bearer token from an ``Authorization`` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


async def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> None:
    """Validate the ``Authorization: Bearer`` header against the configured API key."""
    expected = settings.api_key.get_secret_value() if settings else ""
    if not expected:
        return

    provided = _extract_bearer_token(authorization)
    if provided is None:
        logger.warning("auth.missing_bearer_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        logger.warning("auth.invalid_api_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
