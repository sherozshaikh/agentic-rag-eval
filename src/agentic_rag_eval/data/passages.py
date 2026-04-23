from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import Passage

logger = get_logger(__name__)


def passage_id_for(title: str, text: str) -> str:
    """Return a deterministic 16-hex-char SHA-256 prefix of ``title + text``."""
    digest = hashlib.sha256(f"{title}{text}".encode()).hexdigest()
    return digest[:16]


@dataclass
class PassageRecord:
    """A deduplicated HotpotQA paragraph with the question IDs it appears in."""

    passage_id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    question_ids: set[str] = field(default_factory=set)

    def to_passage(self) -> Passage:
        """Project to the public ``schemas.Passage`` model."""
        return Passage(passage_id=self.passage_id, title=self.title, text=self.text)

    def to_row(self) -> dict[str, Any]:
        """Return a parquet-ready dict representation."""
        return {
            "passage_id": self.passage_id,
            "title": self.title,
            "text": self.text,
            "sentences": list(self.sentences),
            "question_ids": sorted(self.question_ids),
            "num_questions": len(self.question_ids),
        }


def _iter_context_paragraphs(
    context: Any,
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(title, sentences)`` pairs from a HotpotQA context cell."""
    if context is None:
        return
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences_list = context.get("sentences") or []
        for title, sentences in zip(titles, sentences_list, strict=False):
            yield str(title), [str(s) for s in (sentences or [])]
        return
    if isinstance(context, list | tuple):
        for entry in context:
            if not entry:
                continue
            if isinstance(entry, dict):
                title = entry.get("title", "")
                sentences = entry.get("sentences") or entry.get("text") or []
                yield str(title), [str(s) for s in sentences]
            elif isinstance(entry, list | tuple) and len(entry) >= 2:
                title, sentences = entry[0], entry[1]
                if sentences is None:
                    sentences = []
                yield str(title), [str(s) for s in sentences]
        return
    raise TypeError(f"Unsupported context type: {type(context).__name__}")


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s is not None).strip()


def extract_unique_passages(
    dataset: pd.DataFrame | Iterable[dict[str, Any]],
) -> list[PassageRecord]:
    """Flatten HotpotQA contexts into a deduplicated list sorted by passage_id."""
    rows: Iterable[dict[str, Any]]
    if isinstance(dataset, pd.DataFrame):
        rows = dataset.to_dict(orient="records")
    else:
        rows = dataset

    records: dict[str, PassageRecord] = {}
    row_count = 0
    skipped_rows = 0

    for row in rows:
        row_count += 1
        context = row.get("context")
        if context is None:
            skipped_rows += 1
            continue

        qid = row.get("_id") or row.get("question_id")
        qid = str(qid) if qid is not None else None

        try:
            paragraphs = list(_iter_context_paragraphs(context))
        except TypeError as exc:
            logger.warning(
                "Skipping row with unsupported context shape",
                extra={"question_id": qid, "error": str(exc)},
            )
            skipped_rows += 1
            continue

        for title, sentences in paragraphs:
            text = _join_sentences(sentences)
            if not text:
                continue
            pid = passage_id_for(title, text)
            rec = records.get(pid)
            if rec is None:
                rec = PassageRecord(
                    passage_id=pid,
                    title=title,
                    text=text,
                    sentences=list(sentences),
                )
                records[pid] = rec
            if qid is not None:
                rec.question_ids.add(qid)

    logger.info(
        "Extracted unique passages",
        extra={
            "rows_seen": row_count,
            "rows_skipped": skipped_rows,
            "unique_passages": len(records),
        },
    )

    return [records[pid] for pid in sorted(records)]


def passages_to_dataframe(passages: Iterable[PassageRecord]) -> pd.DataFrame:
    """Return passages as a parquet-ready DataFrame."""
    return pd.DataFrame([p.to_row() for p in passages])
