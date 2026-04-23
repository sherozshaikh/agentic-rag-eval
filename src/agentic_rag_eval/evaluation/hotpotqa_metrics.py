from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable
from typing import Any

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, remove articles, and collapse whitespace (HotpotQA/SQuAD rules)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if s is None:
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized prediction equals normalized gold, else 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def _f1_components(pred: str, gold: str) -> tuple[float, float, float]:
    """Return (f1, precision, recall) for a single pair."""
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)

    zero = (0.0, 0.0, 0.0)
    one = (1.0, 1.0, 1.0)

    if normalized_pred in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_pred != normalized_gold:
        return zero

    pred_tokens = normalized_pred.split()
    gold_tokens = normalized_gold.split()

    if not pred_tokens and not gold_tokens:
        return one
    if not pred_tokens or not gold_tokens:
        return zero

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    f1, _p, _r = _f1_components(pred, gold)
    return f1


def _coerce_fact(fact: Any) -> tuple[str, int | None]:
    """Normalize a supporting fact to a ``(title, sent_id)`` tuple."""
    if isinstance(fact, str):
        return fact, None
    if isinstance(fact, dict):
        return str(fact.get("title", "")), fact.get("sent_id")
    if isinstance(fact, list | tuple):
        title = str(fact[0]) if fact else ""
        sent_id = fact[1] if len(fact) > 1 else None
        return title, sent_id
    return str(fact), None


def supporting_fact_f1(
    pred_titles: Iterable[str],
    gold_facts: Iterable[Any],
) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` over supporting-fact titles (title-set level)."""
    pred_set = {normalize_answer(t) for t in pred_titles if t}
    gold_set = {normalize_answer(_coerce_fact(f)[0]) for f in gold_facts}
    gold_set.discard("")
    pred_set.discard("")

    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1
