from __future__ import annotations

from typing import Any

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.evaluation.hotpotqa_metrics import exact_match, f1_score
from agentic_rag_eval.evaluation.judge import _extract_json
from agentic_rag_eval.llm import LLMClient, build_llm_client
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.prompts import get_prompt_registry
from agentic_rag_eval.schemas import EvalRecord, FailureMode, LLMMessage, QueryResponse

logger = get_logger(__name__)


class FailureClassifier:
    """Classify why a prediction failed, via the evaluation LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        eval_llm: LLMClient | None = None,
        f1_threshold: float = 0.5,
    ) -> None:
        self._settings = settings or get_settings()
        self._eval_llm = eval_llm or build_llm_client(self._settings, role="eval")
        self._registry = get_prompt_registry()
        self._f1_threshold = f1_threshold

    def classify(
        self,
        record: EvalRecord,
        predicted_response: QueryResponse | None = None,
    ) -> FailureMode:
        """Return the failure mode for a record, or ``FailureMode.NONE`` on success."""
        em = record.exact_match
        f1 = record.f1
        if em == 0.0 and f1 == 0.0:
            em = exact_match(record.predicted_answer, record.gold_answer)
            f1 = f1_score(record.predicted_answer, record.gold_answer)

        is_failure = em == 0.0 or f1 < self._f1_threshold
        if not is_failure:
            return FailureMode.NONE

        sub_questions = self._format_sub_questions(predicted_response)
        retrieved_titles = self._format_retrieved_titles(predicted_response, record)
        gold_titles = self._format_gold_titles(record)

        try:
            prompt = self._registry.get("failure_classify").render(
                question=record.question,
                gold_answer=record.gold_answer,
                predicted_answer=record.predicted_answer,
                sub_questions=sub_questions,
                retrieved_titles=retrieved_titles,
                gold_titles=gold_titles,
            )
        except KeyError as e:
            logger.warning("failure_classify_prompt_missing", extra={"error": str(e)})
            return FailureMode.NONE

        try:
            result = self._eval_llm.complete(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("failure_classify_llm_failed", extra={"error": str(e)})
            return FailureMode.NONE

        data = _extract_json(result.content)
        if data is None:
            logger.warning(
                "failure_classify_parse_failed",
                extra={"content": result.content[:500]},
            )
            return FailureMode.NONE

        raw = str(data.get("failure_mode", "")).strip().lower()
        try:
            return FailureMode(raw)
        except ValueError:
            logger.warning("failure_classify_unknown_mode", extra={"raw": raw})
            return FailureMode.NONE

    @staticmethod
    def _format_sub_questions(response: QueryResponse | None) -> str:
        if response is None or not response.sub_questions:
            return "(none)"
        return " | ".join(f"{i + 1}. {sq.text}" for i, sq in enumerate(response.sub_questions))

    @staticmethod
    def _format_retrieved_titles(
        response: QueryResponse | None,
        record: EvalRecord,
    ) -> str:
        titles: list[str] = []
        if response is not None:
            for p in response.evidence:
                if p.title:
                    titles.append(p.title)
        if not titles:
            titles = list(record.retrieved_passage_ids)
        if not titles:
            return "(none)"
        return " | ".join(titles)

    @staticmethod
    def _format_gold_titles(record: EvalRecord) -> str:
        titles: list[str] = []
        for fact in record.gold_supporting_facts:
            if isinstance(fact, dict):
                t: Any = fact.get("title")
            elif isinstance(fact, list | tuple) and fact:
                t = fact[0]
            else:
                t = str(fact)
            if t:
                titles.append(str(t))
        if not titles:
            return "(none)"
        seen: set[str] = set()
        uniq: list[str] = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return " | ".join(uniq)
