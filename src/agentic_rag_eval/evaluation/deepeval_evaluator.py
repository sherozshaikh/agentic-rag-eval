from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.llm import LLMClient, build_llm_client
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import LLMMessage

logger = get_logger(__name__)


def _make_deepeval_llm(client: LLMClient) -> Any:
    """Build a DeepEvalBaseLLM adapter around our LLMClient, or None if deepeval is unavailable."""
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except Exception as e:
        logger.warning("deepeval_base_llm_import_failed", extra={"error": str(e)})
        return None

    class _Wrapper(DeepEvalBaseLLM):
        def __init__(self, inner: LLMClient) -> None:
            self._inner = inner

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            result = self._inner.complete(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
            )
            return result.content

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self._inner.model

    return _Wrapper(client)


_G_EVAL_CRITERION = (
    "Is the answer logically consistent with the retrieved evidence and the question? "
    "Check that the answer is supported by the evidence, does not contradict the question, "
    "and does not introduce facts absent from the evidence."
)


class DeepEvalEvaluator:
    """Score test cases with DeepEval metrics against the isolated eval LLM."""

    METRIC_NAMES = ("g_eval", "hallucination", "answer_relevancy")

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        eval_llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._eval_llm = eval_llm or build_llm_client(self._settings, role="eval")
        self._deepeval_llm = _make_deepeval_llm(self._eval_llm)

    def evaluate(self, test_case: Mapping[str, Any]) -> dict[str, float | None]:
        """Score a single test case and return per-metric scores (None on failure)."""
        result: dict[str, float | None] = {name: None for name in self.METRIC_NAMES}

        if self._deepeval_llm is None:
            return result

        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                GEval,
                HallucinationMetric,
            )
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        except Exception as e:
            logger.warning("deepeval_import_failed", extra={"error": str(e)})
            return result

        question = str(test_case.get("question") or test_case.get("input") or "")
        actual = str(test_case.get("actual_output") or "")
        expected = str(test_case.get("expected_output") or "")
        context = list(test_case.get("retrieval_context") or [])

        try:
            tc = LLMTestCase(
                input=question,
                actual_output=actual,
                expected_output=expected,
                retrieval_context=context,
                context=context,
            )
        except Exception as e:
            logger.warning("deepeval_test_case_failed", extra={"error": str(e)})
            return result

        try:
            g_eval = GEval(
                name="LogicalConsistency",
                criteria=_G_EVAL_CRITERION,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                model=self._deepeval_llm,
            )
            g_eval.measure(tc)
            result["g_eval"] = float(g_eval.score) if g_eval.score is not None else None
        except Exception as e:
            logger.warning("deepeval_g_eval_failed", extra={"error": str(e)})

        try:
            hallu = HallucinationMetric(model=self._deepeval_llm)
            hallu.measure(tc)
            result["hallucination"] = float(hallu.score) if hallu.score is not None else None
        except Exception as e:
            logger.warning("deepeval_hallucination_failed", extra={"error": str(e)})

        try:
            rel = AnswerRelevancyMetric(model=self._deepeval_llm)
            rel.measure(tc)
            result["answer_relevancy"] = float(rel.score) if rel.score is not None else None
        except Exception as e:
            logger.warning("deepeval_answer_relevancy_failed", extra={"error": str(e)})

        return result
