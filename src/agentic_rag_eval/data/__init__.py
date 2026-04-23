from __future__ import annotations

from agentic_rag_eval.data.loader import HotpotQALoader
from agentic_rag_eval.data.passages import (
    PassageRecord,
    extract_unique_passages,
    passage_id_for,
)
from agentic_rag_eval.data.subset import StratifiedSubsetResult, stratified_subset
from agentic_rag_eval.data.validate import ChiSquaredReport, chi_squared_validate

__all__ = [
    "ChiSquaredReport",
    "HotpotQALoader",
    "PassageRecord",
    "StratifiedSubsetResult",
    "chi_squared_validate",
    "extract_unique_passages",
    "passage_id_for",
    "stratified_subset",
]
