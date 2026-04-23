from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from agentic_rag_eval import logging_setup
from agentic_rag_eval.config import Settings
from agentic_rag_eval.logging_setup import ContextFilter, configure_logging


@pytest.fixture(autouse=True)
def _reset_logging_state() -> Iterator[None]:
    """Reset the module-level configure-once flag and root handlers per test."""
    logging_setup._CONFIGURED = False
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in saved_handlers:
        root.removeHandler(h)
    yield

    logging_setup._CONFIGURED = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_configure_logging_is_idempotent() -> None:
    """Calling configure_logging twice must not attach duplicate handlers."""
    settings = Settings(log_format="json", log_level="INFO")
    configure_logging(settings)
    first_handlers = list(logging.getLogger().handlers)

    configure_logging(settings)
    second_handlers = list(logging.getLogger().handlers)

    assert first_handlers == second_handlers
    assert len(second_handlers) == 1


def test_json_format_produces_valid_json_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With log_format='json', each emitted record must be a JSON object."""
    settings = Settings(log_format="json", log_level="INFO")
    configure_logging(settings)

    logger = logging.getLogger("agentic_rag_eval.test_json")
    logger.info("hello_world", extra={"answer": 42})

    captured = capsys.readouterr().out.strip()
    assert captured, "expected JSON log line on stdout"

    line = captured.splitlines()[-1]
    payload = json.loads(line)

    assert payload["message"] == "hello_world"
    assert payload["level"] == "INFO"
    assert payload["answer"] == 42


def test_context_filter_injects_fields() -> None:
    """The filter must attach service/version attributes on records."""
    f = ContextFilter({"service": "svc", "version": "9.9.9"})
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    assert f.filter(record) is True
    assert record.service == "svc"
    assert record.version == "9.9.9"


def test_context_filter_injected_via_configure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After configure_logging, records must carry service/version in JSON output."""
    settings = Settings(log_format="json", log_level="INFO")
    configure_logging(settings)

    logging.getLogger("agentic_rag_eval.test_ctx").info("check_ctx")

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload.get("service") == "agentic-rag-eval"
    assert payload.get("version") == "0.1.0"


def test_context_filter_does_not_overwrite_existing_attrs() -> None:
    """Records that already define a context key must not be clobbered."""
    f = ContextFilter({"service": "default"})
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    record.service = "override"
    f.filter(record)
    assert record.service == "override"
