from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.llm import LLMClient, build_llm_client
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import LLMMessage

logger = get_logger(__name__)


class _LLMClientLangchainAdapter:
    """Minimal LangChain-compatible wrapper around ``LLMClient`` for RAGAS."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def invoke(self, prompt: Any, *_args: Any, **_kwargs: Any) -> str:
        text = self._coerce_prompt(prompt)
        result = self._client.complete(
            [LLMMessage(role="user", content=text)],
            temperature=0.0,
        )
        return result.content

    def __call__(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        return self.invoke(prompt, *args, **kwargs)

    def generate(self, prompts: Sequence[Any], *_args: Any, **_kwargs: Any) -> list[str]:
        return [self.invoke(p) for p in prompts]

    @property
    def model_name(self) -> str:
        return self._client.model

    @staticmethod
    def _coerce_prompt(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if hasattr(prompt, "to_string"):
            try:
                return str(prompt.to_string())
            except Exception:
                pass
        if hasattr(prompt, "text"):
            return str(prompt.text)
        return str(prompt)


def _build_ragas_llm(client: LLMClient) -> Any:
    """Return an object RAGAS can use as its LLM."""
    adapter = _LLMClientLangchainAdapter(client)
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(adapter)
    except Exception as e:
        logger.warning("ragas_langchain_wrapper_unavailable", extra={"error": str(e)})
        return adapter


class RAGASEvaluator:
    """Compute RAGAS metrics against an isolated evaluation LLM."""

    METRIC_NAMES = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        eval_llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._eval_llm = eval_llm or build_llm_client(self._settings, role="eval")

    def evaluate_batch(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        ground_truths: Sequence[str],
    ) -> pd.DataFrame:
        """Score a batch of rows; returns a DataFrame with one column per metric (None on failure)."""
        n = len(questions)
        if not (len(answers) == n == len(contexts) == len(ground_truths)):
            raise ValueError(
                "RAGAS batch inputs must have the same length "
                f"(got {len(questions)}, {len(answers)}, {len(contexts)}, {len(ground_truths)})"
            )

        empty = self._empty_frame(n)
        if n == 0:
            return empty

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except Exception as e:
            logger.warning("ragas_import_failed", extra={"error": str(e)})
            return empty

        try:
            ragas_llm = _build_ragas_llm(self._eval_llm)
        except Exception as e:
            logger.warning("ragas_llm_build_failed", extra={"error": str(e)})
            return empty

        dataset = Dataset.from_dict(
            {
                "question": list(questions),
                "answer": list(answers),
                "contexts": [list(c) for c in contexts],
                "ground_truth": list(ground_truths),
            }
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            result = evaluate(dataset, metrics=metrics, llm=ragas_llm)
        except Exception as e:
            logger.warning("ragas_evaluate_failed", extra={"error": str(e)})
            return empty

        try:
            df = result.to_pandas()
        except Exception as e:
            logger.warning("ragas_to_pandas_failed", extra={"error": str(e)})
            return empty

        out = self._empty_frame(n)
        for name in self.METRIC_NAMES:
            if name in df.columns:
                out[name] = df[name].tolist()
        return out

    @classmethod
    def _empty_frame(cls, n: int) -> pd.DataFrame:
        return pd.DataFrame({name: [None] * n for name in cls.METRIC_NAMES})
