from __future__ import annotations

import json
import re
from typing import Any

from agentic_rag_eval.llm import LLMClient
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.prompts import PromptRegistry, get_prompt_registry
from agentic_rag_eval.schemas import (
    LLMMessage,
    QueryType,
    RetrievalStrategy,
    SubQuestion,
)

logger = get_logger(__name__)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class QueryDecomposer:
    """Decompose a question into typed sub-questions via an LLM."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._registry = prompt_registry or get_prompt_registry()
        self._prompt = self._registry.get("decompose")

    def decompose(self, question: str) -> tuple[QueryType, list[SubQuestion]]:
        """Decompose `question` into a `QueryType` and a non-empty list of `SubQuestion`s."""
        rendered = self._prompt.render(question=question)
        messages = [
            LLMMessage(role="user", content=rendered),
        ]

        try:
            result = self._llm.complete(
                messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(
                "decomposer_llm_call_failed",
                extra={"error": str(e), "question": question[:200]},
            )
            return self._fallback(question)

        raw = (result.content or "").strip()
        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning(
                "decomposer_json_parse_failed",
                extra={"question": question[:200], "raw_response": raw[:500]},
            )
            return self._fallback(question)

        try:
            query_type = self._coerce_query_type(parsed.get("query_type"))
            sub_questions = self._coerce_sub_questions(parsed.get("sub_questions"))
        except Exception as e:
            logger.warning(
                "decomposer_schema_invalid",
                extra={"error": str(e), "parsed": str(parsed)[:500]},
            )
            return self._fallback(question)

        if not sub_questions:
            logger.warning(
                "decomposer_empty_sub_questions",
                extra={"question": question[:200]},
            )
            return self._fallback(question)

        logger.info(
            "decomposition_success",
            extra={
                "query_type": query_type.value,
                "n_sub_questions": len(sub_questions),
                "question": question[:200],
            },
        )
        return query_type, sub_questions

    def _fallback(self, question: str) -> tuple[QueryType, list[SubQuestion]]:
        """Return a trivial single-hop decomposition."""
        sub = SubQuestion(
            text=question,
            strategy=RetrievalStrategy.HYBRID,
            depends_on=[],
        )
        return QueryType.SINGLE_HOP, [sub]

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Parse raw text as a JSON object, tolerating fenced or wrapped output."""
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        match = _JSON_OBJECT_RE.search(raw)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _coerce_query_type(value: Any) -> QueryType:
        if isinstance(value, str):
            try:
                return QueryType(value)
            except ValueError:
                pass
        return QueryType.UNKNOWN

    @staticmethod
    def _coerce_sub_questions(value: Any) -> list[SubQuestion]:
        if not isinstance(value, list):
            return []

        out: list[SubQuestion] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            strategy_raw = item.get("strategy", "hybrid")
            try:
                strategy = RetrievalStrategy(strategy_raw)
            except ValueError:
                strategy = RetrievalStrategy.HYBRID

            depends_raw = item.get("depends_on", []) or []
            depends_on: list[int] = []
            if isinstance(depends_raw, list):
                for d in depends_raw:
                    if isinstance(d, int):
                        depends_on.append(d)
                    elif isinstance(d, str) and d.isdigit():
                        depends_on.append(int(d))

            out.append(
                SubQuestion(
                    text=text.strip(),
                    strategy=strategy,
                    depends_on=depends_on,
                )
            )
        return out
