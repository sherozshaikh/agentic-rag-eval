from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from agentic_rag_eval.data.passages import (
    PassageRecord,
    extract_unique_passages,
    passage_id_for,
    passages_to_dataframe,
)
from agentic_rag_eval.schemas import Passage


def test_passage_id_is_16_hex_chars() -> None:
    pid = passage_id_for("Apollo 11", "Apollo 11 was the first crewed mission.")
    assert len(pid) == 16
    assert all(c in "0123456789abcdef" for c in pid)


def test_passage_id_is_deterministic() -> None:
    a = passage_id_for("Title", "Some body text.")
    b = passage_id_for("Title", "Some body text.")
    assert a == b


def test_passage_id_differs_for_different_titles() -> None:
    a = passage_id_for("Title A", "Same body.")
    b = passage_id_for("Title B", "Same body.")
    assert a != b


def test_passage_id_differs_for_different_text() -> None:
    a = passage_id_for("Same Title", "body one")
    b = passage_id_for("Same Title", "body two")
    assert a != b


def test_passage_id_matches_sha256_prefix() -> None:
    title, text = "Foo", "Bar"
    expected = hashlib.sha256(f"{title}{text}".encode()).hexdigest()[:16]
    assert passage_id_for(title, text) == expected


def _row(qid: str, context: object) -> dict[str, object]:
    return {"_id": qid, "context": context}


def test_extract_from_column_layout_dict() -> None:
    context = {
        "title": ["Article A", "Article B"],
        "sentences": [
            ["First sentence.", " Second sentence."],
            ["Only sentence."],
        ],
    }
    rows = [_row("q1", context)]
    passages = extract_unique_passages(rows)
    assert len(passages) == 2
    titles = {p.title for p in passages}
    assert titles == {"Article A", "Article B"}


def test_extract_from_row_layout_list() -> None:
    context = [
        ["Article A", ["sentence 1.", " sentence 2."]],
        ["Article B", ["sentence 1."]],
    ]
    rows = [_row("q1", context)]
    passages = extract_unique_passages(rows)
    assert len(passages) == 2


def test_extract_deduplicates_identical_paragraphs_across_questions() -> None:
    context_a = {"title": ["Shared"], "sentences": [["Dupe body text."]]}
    context_b = {"title": ["Shared"], "sentences": [["Dupe body text."]]}
    rows = [_row("q1", context_a), _row("q2", context_b)]
    passages = extract_unique_passages(rows)
    assert len(passages) == 1
    rec = passages[0]
    assert rec.question_ids == {"q1", "q2"}


def test_extract_keeps_different_paragraphs_with_same_title() -> None:
    context = {
        "title": ["Same Title", "Same Title"],
        "sentences": [["First body."], ["Second body."]],
    }
    rows = [_row("q1", context)]
    passages = extract_unique_passages(rows)
    assert len(passages) == 2


def test_extract_skips_empty_context() -> None:
    rows = [_row("q1", None), _row("q2", {"title": [], "sentences": []})]
    passages = extract_unique_passages(rows)
    assert passages == []


def test_extract_skips_paragraphs_with_no_text() -> None:
    context = {
        "title": ["A", "B"],
        "sentences": [[], ["   "]],
    }
    rows = [_row("q1", context)]
    passages = extract_unique_passages(rows)
    assert passages == []


def test_extract_is_deterministic_and_sorted_by_id() -> None:
    context = {
        "title": ["Z", "A", "M"],
        "sentences": [["z body"], ["a body"], ["m body"]],
    }
    rows = [_row("q1", context)]
    passages = extract_unique_passages(rows)
    ids = [p.passage_id for p in passages]
    assert ids == sorted(ids)


def test_extract_records_question_ids_per_passage() -> None:
    ctx1 = {
        "title": ["A", "B"],
        "sentences": [["a body"], ["b body"]],
    }
    ctx2 = {
        "title": ["B", "C"],
        "sentences": [["b body"], ["c body"]],
    }
    rows = [_row("q1", ctx1), _row("q2", ctx2)]
    passages = extract_unique_passages(rows)
    by_title = {p.title: p for p in passages}
    assert by_title["A"].question_ids == {"q1"}
    assert by_title["B"].question_ids == {"q1", "q2"}
    assert by_title["C"].question_ids == {"q2"}


def test_extract_accepts_dataframe_input() -> None:
    df = pd.DataFrame(
        [
            _row(
                "q1",
                {"title": ["A"], "sentences": [["a body"]]},
            ),
            _row(
                "q2",
                {"title": ["A"], "sentences": [["a body"]]},
            ),
        ]
    )
    passages = extract_unique_passages(df)
    assert len(passages) == 1
    assert passages[0].question_ids == {"q1", "q2"}


def test_extract_handles_question_id_alias() -> None:
    rows = [{"question_id": "qX", "context": {"title": ["T"], "sentences": [["body"]]}}]
    passages = extract_unique_passages(rows)
    assert passages[0].question_ids == {"qX"}


def test_extract_skips_row_with_unsupported_context_type() -> None:
    rows = [_row("q1", 12345)]

    passages = extract_unique_passages(rows)
    assert passages == []


def test_passage_record_to_passage_returns_schema_passage() -> None:
    rec = PassageRecord(
        passage_id="abc123",
        title="T",
        text="body",
        sentences=["body"],
        question_ids={"q1"},
    )
    p = rec.to_passage()
    assert isinstance(p, Passage)
    assert p.passage_id == "abc123"
    assert p.title == "T"
    assert p.text == "body"


def test_passage_record_to_row_is_serializable() -> None:
    import json

    rec = PassageRecord(
        passage_id="abc",
        title="T",
        text="body",
        sentences=["body"],
        question_ids={"q2", "q1"},
    )
    row = rec.to_row()

    encoded = json.dumps(row)
    decoded = json.loads(encoded)
    assert decoded["passage_id"] == "abc"
    assert decoded["question_ids"] == ["q1", "q2"]
    assert decoded["num_questions"] == 2


def test_passages_to_dataframe_columns() -> None:
    recs = [
        PassageRecord(
            passage_id="p1",
            title="T1",
            text="body 1",
            sentences=["body 1"],
            question_ids={"q1"},
        ),
        PassageRecord(
            passage_id="p2",
            title="T2",
            text="body 2",
            sentences=["body 2"],
            question_ids={"q2"},
        ),
    ]
    df = passages_to_dataframe(recs)
    assert set(["passage_id", "title", "text", "sentences", "question_ids"]).issubset(df.columns)
    assert len(df) == 2


@pytest.mark.parametrize(
    "title,text",
    [
        ("", "body"),
        ("title", ""),
        ("unicode ñ", "тело text"),
    ],
)
def test_passage_id_accepts_edge_case_inputs(title: str, text: str) -> None:
    pid = passage_id_for(title, text)
    assert len(pid) == 16
