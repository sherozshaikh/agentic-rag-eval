from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from agentic_rag_eval.config import Settings, get_settings

_CONFIGURED = False


class ContextFilter(logging.Filter):
    """Inject static context fields into every log record."""

    def __init__(self, context: dict[str, Any]) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def configure_logging(settings: Settings | None = None) -> None:
    """Configure the root logger. Idempotent across calls."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = settings or get_settings()

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    fmt: logging.Formatter
    if settings.log_format == "json":
        fmt = JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    handler.setFormatter(fmt)
    handler.addFilter(ContextFilter({"service": "agentic-rag-eval", "version": "0.1.0"}))
    root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
