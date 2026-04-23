from __future__ import annotations

import re
import unicodedata

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)

MAX_QUESTION_LENGTH = 2000
MIN_QUESTION_LENGTH = 1

_SUSPICIOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I),
    re.compile(r"system\s*prompt\s*(is|was|:)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"<\|(?:im_start|im_end|system|user|assistant)\|>", re.I),
    re.compile(r"\bBEGIN\s+NEW\s+INSTRUCTIONS?\b", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
)


class ValidationError(ValueError):
    """Raised when user input fails validation."""


def _strip_control_chars(text: str) -> str:
    """Strip control characters except tab, newline, and carriage return."""
    allowed = {"\t", "\n", "\r"}
    return "".join(ch for ch in text if ch in allowed or unicodedata.category(ch)[0] != "C")


def _detect_suspicious(text: str) -> list[str]:
    """Return matched suspicious-pattern strings."""
    hits: list[str] = []
    for pat in _SUSPICIOUS_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def validate_question(
    text: str,
    *,
    max_length: int = MAX_QUESTION_LENGTH,
    min_length: int = MIN_QUESTION_LENGTH,
) -> str:
    """Sanitize and length-check a user question; raises ``ValidationError`` on failure."""
    if not isinstance(text, str):
        raise ValidationError("question must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    cleaned = _strip_control_chars(normalized).strip()

    if len(cleaned) < min_length:
        raise ValidationError(f"question must be at least {min_length} character(s)")
    if len(cleaned) > max_length:
        raise ValidationError(
            f"question must be at most {max_length} characters (got {len(cleaned)})"
        )

    suspicious = _detect_suspicious(cleaned)
    if suspicious:
        logger.warning(
            "validators.suspicious_input",
            extra={
                "matched_patterns": suspicious,
                "question_preview": cleaned[:200],
            },
        )

    return cleaned
