from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentic_rag_eval.llm.client import LLMClient, LLMClientError
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import (
    LLMMessage,
    Passage,
    QueryResponse,
    QueryType,
    RetrievalResult,
    RetrievalStrategy,
)

if TYPE_CHECKING:
    from agentic_rag_eval.config import Settings
    from agentic_rag_eval.retrieval import Retriever
    from agentic_rag_eval.tracing import TraceLogger
    from agentic_rag_eval.tracing.logger import TraceContext

logger = get_logger(__name__)


_BASELINE_SYSTEM_PROMPT = "Answer the question using only the provided context. Be brief."

_BASELINE_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _format_context(passages: list[Passage]) -> str:
    """Render passages as a numbered context block."""
    if not passages:
        return "(no passages retrieved)"

    lines: list[str] = []
    for idx, passage in enumerate(passages, start=1):
        header = f"[{idx}]"
        if passage.title:
            header = f"{header} {passage.title}"
        lines.append(f"{header}\n{passage.text}")
    return "\n\n".join(lines)


class BaselineRAG:
    """Naive single-shot dense RAG pipeline. Stateless across calls."""

    PIPELINE_NAME: str = "baseline"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        settings: Settings,
        *,
        tracer: TraceLogger | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._settings = settings
        self._tracer = tracer

    def answer(
        self,
        question: str,
        top_k: int = 10,
        *,
        eval_run_id: str | None = None,
    ) -> QueryResponse:
        """Answer ``question`` via a single dense retrieval and LLM call.

        Raises:
            ValueError: If ``question`` is empty or ``top_k`` is not positive.
            LLMClientError: If the LLM call fails after retries.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        question = question.strip()
        pipeline_start = time.perf_counter()

        if self._tracer is not None:
            with self._tracer.trace(
                question,
                pipeline=self.PIPELINE_NAME,
                eval_run_id=eval_run_id,
            ) as ctx:
                response = self._run(question, top_k, trace_ctx=ctx)
                ctx.strategy_used = RetrievalStrategy.DENSE.value
                ctx.confidence = response.confidence
                ctx.failure_mode = "none"
                return response

        response = self._run(question, top_k, trace_ctx=None)
        response.latency_ms = (time.perf_counter() - pipeline_start) * 1000.0
        return response

    def _run(
        self,
        question: str,
        top_k: int,
        *,
        trace_ctx: TraceContext | None,
    ) -> QueryResponse:
        """Execute the single-shot pipeline."""
        start = time.perf_counter()

        retrieval = self._retrieve(question, top_k, trace_ctx=trace_ctx)
        passages = list(retrieval.passages)

        llm_call, prompt_text = self._synthesize(question, passages, trace_ctx=trace_ctx)

        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "baseline_answer_complete",
            extra={
                "pipeline": self.PIPELINE_NAME,
                "num_passages": len(passages),
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": llm_call.prompt_tokens,
                "completion_tokens": llm_call.completion_tokens,
            },
        )

        return QueryResponse(
            answer=llm_call.content.strip(),
            confidence="high",
            query_type=QueryType.SINGLE_HOP,
            sub_questions=[],
            reasoning_chain=[],
            evidence=passages,
            evidence_conflict=False,
            reason=None,
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": llm_call.prompt_tokens,
                "completion_tokens": llm_call.completion_tokens,
                "total_tokens": llm_call.total_tokens,
            },
            cost_usd=llm_call.cost_usd,
            trace_id=trace_ctx.trace_id if trace_ctx is not None else None,
        )

    def _retrieve(
        self,
        question: str,
        top_k: int,
        *,
        trace_ctx: TraceContext | None,
    ) -> RetrievalResult:
        """Run a single dense retrieval, optionally inside a trace span."""
        if self._tracer is not None and trace_ctx is not None:
            with self._tracer.span(trace_ctx, "retrieve") as span:
                span.metadata["strategy"] = RetrievalStrategy.DENSE.value
                span.metadata["top_k"] = top_k
                result = self._retriever.retrieve(
                    question, strategy=RetrievalStrategy.DENSE, top_k=top_k
                )
                span.metadata["returned"] = len(result.passages)
                return result

        return self._retriever.retrieve(question, strategy=RetrievalStrategy.DENSE, top_k=top_k)

    def _synthesize(
        self,
        question: str,
        passages: list[Passage],
        *,
        trace_ctx: TraceContext | None,
    ) -> tuple[object, str]:
        """Call the LLM with the baseline prompt and return ``(result, prompt)``."""
        context_block = _format_context(passages)
        user_prompt = _BASELINE_USER_TEMPLATE.format(context=context_block, question=question)
        messages = [
            LLMMessage(role="system", content=_BASELINE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        def _call() -> object:
            return self._llm.complete(messages, temperature=0.0)

        if self._tracer is not None and trace_ctx is not None:
            with self._tracer.span(trace_ctx, "llm") as span:
                try:
                    call = _call()
                except LLMClientError:
                    span.metadata["error"] = "llm_client_error"
                    raise
                span.metadata["model"] = getattr(call, "model", "")
                span.metadata["backend"] = getattr(call, "backend", "")

            self._tracer.record_llm_call(
                trace_ctx,
                role="baseline",
                call=call,
                prompt=user_prompt,
                response=getattr(call, "content", "") or "",
            )
            return call, user_prompt

        call = _call()
        return call, user_prompt
