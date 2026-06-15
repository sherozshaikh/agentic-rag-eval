from __future__ import annotations

import json
import re
import time
from typing import Any, TypedDict

from agentic_rag_eval.agent.decomposer import QueryDecomposer
from agentic_rag_eval.agent.memory import MemoryStore
from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.llm import LLMClient
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.prompts import PromptRegistry, get_prompt_registry
from agentic_rag_eval.schemas import (
    LLMMessage,
    Passage,
    QueryResponse,
    QueryType,
    ReasoningStep,
    RetrievalStrategy,
    SubQuestion,
)
from agentic_rag_eval.tracing import TraceLogger, get_trace_logger
from agentic_rag_eval.tracing.logger import TraceContext

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Mutable state passed between agent nodes."""

    question: str
    user_id: str | None
    query_type: QueryType
    sub_questions: list[SubQuestion]
    memory_snippets: list[str]
    evidence: list[Passage]
    reasoning_chain: list[ReasoningStep]
    step: int
    context_tokens: int
    budget_exhausted: bool
    budget_exhausted_step: int | None
    answer: str
    confidence: str
    evidence_conflict: bool
    reason: str | None
    error: str | None


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _approx_tokens(text: str) -> int:
    """Cheap deterministic token estimate (roughly 4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


_DEFAULT_TOP_K = 10
_DEFAULT_RERANK_TOP_K = 5


class ReActAgent:
    """ReAct agent that decomposes, retrieves, and reasons over evidence."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Any,
        memory: MemoryStore | None,
        trace_logger: TraceLogger | None = None,
        settings: Settings | None = None,
        *,
        decomposer: QueryDecomposer | None = None,
        prompt_registry: PromptRegistry | None = None,
        top_k: int = _DEFAULT_TOP_K,
        rerank_top_k: int = _DEFAULT_RERANK_TOP_K,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._memory = memory
        self._settings = settings or get_settings()
        self._tracer = trace_logger or get_trace_logger(self._settings)
        self._registry = prompt_registry or get_prompt_registry()
        self._decomposer = decomposer or QueryDecomposer(llm=llm, prompt_registry=self._registry)
        self._reason_prompt = self._registry.get("reason")
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k

        self._graph = self._build_graph()

    def answer(
        self,
        question: str,
        user_id: str | None = None,
        trace_ctx: TraceContext | None = None,
    ) -> QueryResponse:
        """Run the ReAct pipeline and return a populated `QueryResponse`."""
        if trace_ctx is not None:
            return self._run_with_ctx(question, user_id, trace_ctx)

        with self._tracer.trace(question, pipeline="agentic_phase2") as ctx:
            return self._run_with_ctx(question, user_id, ctx)

    def _run_with_ctx(
        self,
        question: str,
        user_id: str | None,
        ctx: TraceContext,
    ) -> QueryResponse:
        start = time.perf_counter()
        state: AgentState = {
            "question": question,
            "user_id": user_id,
            "sub_questions": [],
            "memory_snippets": [],
            "evidence": [],
            "reasoning_chain": [],
            "step": 0,
            "context_tokens": 0,
            "budget_exhausted": False,
            "budget_exhausted_step": None,
            "answer": "",
            "confidence": "high",
            "evidence_conflict": False,
            "reason": None,
            "error": None,
        }

        final_state: AgentState = self._graph.invoke(state, ctx=ctx)

        latency_ms = (time.perf_counter() - start) * 1000.0
        response = self._build_response(final_state, ctx, latency_ms)

        ctx.confidence = response.confidence
        if response.sub_questions:
            ctx.strategy_used = ",".join(sq.strategy.value for sq in response.sub_questions)
        if final_state.get("budget_exhausted"):
            ctx.failure_mode = "context_overflow"

        if self._memory is not None and user_id:
            try:
                self._memory.add_from_response(user_id, question, response)
            except Exception as e:
                logger.warning("memory_add_from_response_failed", extra={"error": str(e)})

        return response

    def _build_graph(self) -> Any:
        """Build a LangGraph StateGraph, or an in-process fallback."""
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            logger.info("langgraph_unavailable_using_fallback")
            return _FallbackGraph(self)

        graph = StateGraph(dict)
        graph.add_node("decompose", self._node_decompose)
        graph.add_node("memory_lookup", self._node_memory_lookup)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("reason", self._node_reason)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "memory_lookup")
        graph.add_edge("memory_lookup", "retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", END)

        compiled = graph.compile()
        return _LangGraphAdapter(compiled, self)

    def _node_decompose(self, state: AgentState, ctx: TraceContext) -> AgentState:
        with self._tracer.span(ctx, "decompose") as span:
            # Ablation: skip LLM decomposition, treat raw question as single sub-question
            if self._settings.ablation_no_decomp:
                query_type, sub_questions = self._decomposer._fallback(state["question"])
                span.metadata["ablation_no_decomp"] = True
            else:
                try:
                    query_type, sub_questions = self._decomposer.decompose(state["question"])
                except Exception as e:
                    logger.warning("decompose_node_error", extra={"error": str(e)})
                    query_type = QueryType.SINGLE_HOP
                    sub_questions = [
                        SubQuestion(
                            text=state["question"],
                            strategy=RetrievalStrategy.HYBRID,
                        )
                    ]
                    state["error"] = f"decompose_failed: {e}"

            state["query_type"] = query_type
            state["sub_questions"] = sub_questions
            span.metadata["query_type"] = query_type.value
            span.metadata["n_sub_questions"] = len(sub_questions)
        return state

    def _node_memory_lookup(self, state: AgentState, ctx: TraceContext) -> AgentState:
        snippets: list[str] = []
        with self._tracer.span(ctx, "memory") as span:
            user_id = state.get("user_id")
            if self._memory is None or not user_id:
                span.metadata["skipped"] = True
                state["memory_snippets"] = snippets
                return state

            try:
                results = self._memory.search(
                    user_id=user_id,
                    query=state["question"],
                    limit=5,
                )
            except Exception as e:
                logger.warning("memory_lookup_failed", extra={"error": str(e)})
                results = []

            for r in results:
                text = r.get("memory") or r.get("text") or r.get("content")
                if isinstance(text, str) and text.strip():
                    snippets.append(text.strip())

            span.metadata["n_snippets"] = len(snippets)

        state["memory_snippets"] = snippets
        return state

    def _node_retrieve(self, state: AgentState, ctx: TraceContext) -> AgentState:
        """Retrieve evidence for each sub-question, respecting step and token budgets."""
        sub_questions = state.get("sub_questions") or []
        max_steps = self._settings.max_agent_steps
        budget = self._settings.effective_context_budget

        ordered = self._topological_order(sub_questions)
        reasoning_chain: list[ReasoningStep] = state.get("reasoning_chain") or []
        evidence: list[Passage] = list(state.get("evidence") or [])
        context_tokens = state.get("context_tokens", 0)
        seen_ids: set[str] = {p.passage_id for p in evidence}

        for i, sq_index in enumerate(ordered):
            if i >= max_steps:
                logger.info(
                    "max_agent_steps_reached",
                    extra={"max_steps": max_steps, "step": i},
                )
                break

            sub = sub_questions[sq_index]
            step_num = i + 1
            with self._tracer.span(ctx, "retrieve") as span:
                span.metadata["step"] = step_num
                span.metadata["sub_question"] = sub.text[:200]
                span.metadata["strategy"] = sub.strategy.value

                try:
                    passages = self._retriever.retrieve_and_rerank(
                        query=sub.text,
                        top_k=self._top_k,
                        rerank_top_k=self._rerank_top_k,
                    )
                except Exception as e:
                    logger.warning(
                        "retrieval_failed",
                        extra={"error": str(e), "sub_question": sub.text[:200]},
                    )
                    passages = []

                span.metadata["n_passages"] = len(passages)

            added_passages: list[Passage] = []
            truncated = False
            for p in passages:
                if p.passage_id in seen_ids:
                    continue
                ptoks = _approx_tokens(p.text)
                if context_tokens + ptoks > budget:
                    truncated = True
                    break
                evidence.append(p)
                added_passages.append(p)
                seen_ids.add(p.passage_id)
                context_tokens += ptoks

            reasoning_chain.append(
                ReasoningStep(
                    step=step_num,
                    thought=(f"Decomposed sub-question [{sub.strategy.value}]: {sub.text}"),
                    action=f"retrieve_and_rerank(strategy={sub.strategy.value})",
                    observation=(
                        f"Retrieved {len(added_passages)} passage(s)"
                        + (" (context budget reached)" if truncated else "")
                    ),
                    retrieved=added_passages,
                )
            )

            if truncated:
                state["budget_exhausted"] = True
                state["budget_exhausted_step"] = step_num
                logger.info(
                    "context_budget_exhausted",
                    extra={
                        "step": step_num,
                        "context_tokens": context_tokens,
                        "budget": budget,
                    },
                )
                break

        state["evidence"] = evidence
        state["reasoning_chain"] = reasoning_chain
        state["context_tokens"] = context_tokens
        state["step"] = len(reasoning_chain)
        return state

    def _node_reason(self, state: AgentState, ctx: TraceContext) -> AgentState:
        """Synthesize a final answer from accumulated evidence."""
        evidence = state.get("evidence") or []
        memory_snippets = state.get("memory_snippets") or []

        evidence_block = self._format_evidence(evidence)
        memory_block = self._format_memory(memory_snippets)

        rendered = self._reason_prompt.render(
            question=state["question"],
            evidence=evidence_block or "(none)",
            memory=memory_block or "(none)",
        )
        messages = [LLMMessage(role="user", content=rendered)]

        with self._tracer.span(ctx, "synthesize") as span:
            try:
                result = self._llm.complete(
                    messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.warning("reason_llm_call_failed", extra={"error": str(e)})
                state["answer"] = ""
                state["confidence"] = "low"
                state["reason"] = f"LLM synthesis failed: {e}"
                span.metadata["error"] = str(e)
                return state

            span.metadata["latency_ms"] = result.latency_ms
            span.metadata["tokens"] = result.total_tokens

            self._tracer.record_llm_call(
                ctx,
                role="reason",
                call=result,
                prompt=rendered,
                response=result.content,
            )

        parsed = self._parse_json(result.content or "")
        if parsed is None:
            logger.warning(
                "reason_json_parse_failed",
                extra={"raw": (result.content or "")[:500]},
            )
            state["answer"] = (result.content or "").strip()[:500]
            state["confidence"] = "low"
            state["reason"] = "Could not parse reasoning JSON"
            return state

        answer_val = parsed.get("answer")
        state["answer"] = str(answer_val) if answer_val is not None else ""
        state["confidence"] = self._coerce_confidence(parsed.get("confidence"))
        state["evidence_conflict"] = bool(parsed.get("evidence_conflict", False))
        reason_val = parsed.get("reason")
        state["reason"] = str(reason_val) if reason_val else None

        reasoning_chain = list(state.get("reasoning_chain") or [])
        reasoning_chain.append(
            ReasoningStep(
                step=len(reasoning_chain) + 1,
                thought=str(parsed.get("reasoning") or "Synthesized final answer"),
                action="synthesize",
                observation=(
                    f"answer={state['answer']!r} confidence={state['confidence']}"
                    + (" conflict=true" if state["evidence_conflict"] else "")
                ),
                retrieved=[],
            )
        )
        state["reasoning_chain"] = reasoning_chain

        if state.get("budget_exhausted"):
            step_n = state.get("budget_exhausted_step") or state.get("step") or 0
            state["confidence"] = "low"
            state["reason"] = f"Context budget exhausted at step {step_n}"

        return state

    @staticmethod
    def _format_evidence(evidence: list[Passage]) -> str:
        if not evidence:
            return ""
        lines: list[str] = []
        for i, p in enumerate(evidence, start=1):
            score = p.rerank_score if p.rerank_score is not None else p.score
            title = p.title or "(untitled)"
            lines.append(f"[{i}] title={title!r} score={score:.3f}\n{p.text}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_memory(snippets: list[str]) -> str:
        if not snippets:
            return ""
        return "\n".join(f"- {s}" for s in snippets)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        raw = (raw or "").strip()
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
    def _coerce_confidence(value: Any) -> str:
        if isinstance(value, str) and value.lower() in {"high", "medium", "low"}:
            return value.lower()
        return "medium"

    @staticmethod
    def _topological_order(sub_questions: list[SubQuestion]) -> list[int]:
        """Return indices in dependency order; falls back to input order on cycle."""
        n = len(sub_questions)
        if n == 0:
            return []

        in_order: list[int] = []
        visited: set[int] = set()

        def visit(i: int, stack: set[int]) -> bool:
            if i in visited:
                return True
            if i in stack:
                return False
            stack.add(i)
            for dep in sub_questions[i].depends_on:
                if 0 <= dep < n:
                    if not visit(dep, stack):
                        return False
            stack.remove(i)
            visited.add(i)
            in_order.append(i)
            return True

        for i in range(n):
            if not visit(i, set()):
                return list(range(n))

        return in_order

    def _build_response(
        self,
        state: AgentState,
        ctx: TraceContext,
        latency_ms: float,
    ) -> QueryResponse:
        return QueryResponse(
            answer=state.get("answer", "") or "",
            confidence=state.get("confidence", "medium"),
            query_type=state.get("query_type", QueryType.UNKNOWN),
            sub_questions=state.get("sub_questions") or [],
            reasoning_chain=state.get("reasoning_chain") or [],
            evidence=state.get("evidence") or [],
            evidence_conflict=bool(state.get("evidence_conflict", False)),
            reason=state.get("reason"),
            latency_ms=latency_ms,
            token_usage={
                "prompt_tokens": ctx.prompt_tokens,
                "completion_tokens": ctx.completion_tokens,
                "total_tokens": ctx.total_tokens,
            },
            cost_usd=ctx.cost_usd,
            trace_id=ctx.trace_id,
        )


class _LangGraphAdapter:
    """Adapts a compiled LangGraph to the `.invoke(state, ctx=...)` API."""

    def __init__(self, compiled: Any, agent: ReActAgent) -> None:
        self._compiled = compiled
        self._agent = agent

    def invoke(self, state: AgentState, *, ctx: TraceContext) -> AgentState:
        _ = self._compiled
        return _FallbackGraph(self._agent).invoke(state, ctx=ctx)


class _FallbackGraph:
    """Sequential in-process replacement for a LangGraph StateGraph."""

    def __init__(self, agent: ReActAgent) -> None:
        self._agent = agent

    def invoke(self, state: AgentState, *, ctx: TraceContext) -> AgentState:
        state = self._agent._node_decompose(state, ctx)
        state = self._agent._node_memory_lookup(state, ctx)
        state = self._agent._node_retrieve(state, ctx)
        state = self._agent._node_reason(state, ctx)
        return state
