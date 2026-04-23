from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentic_rag_eval.baseline import BaselineRAG
from agentic_rag_eval.baseline.naive_rag import _BASELINE_SYSTEM_PROMPT, _format_context
from agentic_rag_eval.llm.client import LLMClientError
from agentic_rag_eval.schemas import (
    LLMCallResult,
    LLMMessage,
    Passage,
    QueryType,
    RetrievalResult,
    RetrievalStrategy,
)


@dataclass
class _FakeLLMClient:
    """Minimal stand-in for :class:`LLMClient`.

    Records the messages it was called with and returns a canned
    :class:`LLMCallResult`. Set ``raise_exc`` to simulate failure after
    retries are exhausted.
    """

    response_content: str = "Paris is the capital of France."
    prompt_tokens: int = 42
    completion_tokens: int = 7
    raise_exc: Exception | None = None
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    model: str = "fake-model"
    backend: str = "local"

    def complete(
        self,
        messages: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        payload: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, LLMMessage):
                payload.append({"role": m.role, "content": m.content})
            else:
                payload.append(dict(m))
        self.calls.append(payload)

        if self.raise_exc is not None:
            raise self.raise_exc

        return LLMCallResult(
            content=self.response_content,
            model=self.model,
            backend=self.backend,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            latency_ms=12.5,
            cost_usd=0.0,
            finish_reason="stop",
        )


@dataclass
class _FakeRetriever:
    """Minimal stand-in for :class:`Retriever`."""

    passages: list[Passage] = field(default_factory=list)
    latency_ms: float = 5.0
    calls: list[tuple[str, RetrievalStrategy, int]] = field(default_factory=list)
    raise_exc: Exception | None = None

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int = 10,
    ) -> RetrievalResult:
        self.calls.append((query, strategy, top_k))
        if self.raise_exc is not None:
            raise self.raise_exc
        return RetrievalResult(
            query=query,
            strategy=strategy,
            passages=list(self.passages[:top_k]),
            latency_ms=self.latency_ms,
        )


def _make_passages() -> list[Passage]:
    return [
        Passage(
            passage_id="p1",
            title="France",
            text="France is a country in Western Europe; its capital is Paris.",
            score=0.91,
            source_strategy=RetrievalStrategy.DENSE,
        ),
        Passage(
            passage_id="p2",
            title="Paris",
            text="Paris is the capital and most populous city of France.",
            score=0.87,
            source_strategy=RetrievalStrategy.DENSE,
        ),
    ]


@pytest.fixture()
def fake_llm() -> _FakeLLMClient:
    return _FakeLLMClient()


@pytest.fixture()
def fake_retriever() -> _FakeRetriever:
    return _FakeRetriever(passages=_make_passages())


@pytest.fixture()
def baseline(fake_llm: _FakeLLMClient, fake_retriever: _FakeRetriever) -> BaselineRAG:
    return BaselineRAG(
        llm=fake_llm,
        retriever=fake_retriever,
        settings=object(),
        tracer=None,
    )


class TestFormatContext:
    def test_formats_each_passage_with_index_and_title(self) -> None:
        passages = _make_passages()
        rendered = _format_context(passages)
        assert "[1] France" in rendered
        assert "[2] Paris" in rendered
        assert "France is a country" in rendered
        assert "Paris is the capital" in rendered

    def test_handles_passage_without_title(self) -> None:
        rendered = _format_context([Passage(passage_id="x", title=None, text="hello world")])
        assert rendered.startswith("[1]\nhello world")

    def test_empty_passage_list_returns_placeholder(self) -> None:
        assert _format_context([]) == "(no passages retrieved)"


class TestBaselineRAGAnswerHappyPath:
    def test_returns_llm_answer_with_evidence(
        self,
        baseline: BaselineRAG,
        fake_llm: _FakeLLMClient,
        fake_retriever: _FakeRetriever,
    ) -> None:
        response = baseline.answer("What is the capital of France?", top_k=5)

        assert response.answer == "Paris is the capital of France."
        assert response.confidence == "high"
        assert response.query_type is QueryType.SINGLE_HOP
        assert response.reasoning_chain == []
        assert response.sub_questions == []
        assert response.evidence_conflict is False
        assert response.reason is None

        assert [p.passage_id for p in response.evidence] == ["p1", "p2"]

        assert response.token_usage == {
            "prompt_tokens": 42,
            "completion_tokens": 7,
            "total_tokens": 49,
        }

        assert response.latency_ms > 0.0

        assert fake_retriever.calls == [
            ("What is the capital of France?", RetrievalStrategy.DENSE, 5)
        ]

        assert len(fake_llm.calls) == 1
        payload = fake_llm.calls[0]
        assert payload[0] == {"role": "system", "content": _BASELINE_SYSTEM_PROMPT}
        assert payload[1]["role"] == "user"
        user_content = payload[1]["content"]
        assert "What is the capital of France?" in user_content
        assert "France is a country" in user_content
        assert "Paris is the capital" in user_content

    def test_default_top_k_is_ten(
        self,
        baseline: BaselineRAG,
        fake_retriever: _FakeRetriever,
    ) -> None:
        baseline.answer("anything")
        assert fake_retriever.calls[0][2] == 10

    def test_question_is_stripped(
        self, baseline: BaselineRAG, fake_retriever: _FakeRetriever
    ) -> None:
        baseline.answer("   padded question   ")
        assert fake_retriever.calls[0][0] == "padded question"

    def test_handles_empty_retrieval_gracefully(
        self,
        fake_llm: _FakeLLMClient,
    ) -> None:
        retriever = _FakeRetriever(passages=[])
        pipeline = BaselineRAG(
            llm=fake_llm,
            retriever=retriever,
            settings=object(),
        )

        response = pipeline.answer("lonely question")

        assert response.evidence == []

        assert len(fake_llm.calls) == 1
        user_prompt = fake_llm.calls[0][1]["content"]
        assert "(no passages retrieved)" in user_prompt
        assert response.answer


class TestBaselineRAGErrorHandling:
    def test_empty_question_raises(self, baseline: BaselineRAG) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            baseline.answer("")
        with pytest.raises(ValueError, match="non-empty"):
            baseline.answer("   ")

    def test_non_positive_top_k_raises(self, baseline: BaselineRAG) -> None:
        with pytest.raises(ValueError, match="positive"):
            baseline.answer("q", top_k=0)
        with pytest.raises(ValueError, match="positive"):
            baseline.answer("q", top_k=-1)

    def test_llm_failure_propagates(
        self,
        fake_retriever: _FakeRetriever,
    ) -> None:
        failing_llm = _FakeLLMClient(raise_exc=LLMClientError("backend unreachable"))
        pipeline = BaselineRAG(
            llm=failing_llm,
            retriever=fake_retriever,
            settings=object(),
        )
        with pytest.raises(LLMClientError, match="backend unreachable"):
            pipeline.answer("any question")

        assert len(fake_retriever.calls) == 1

    def test_retriever_exception_propagates(
        self,
        fake_llm: _FakeLLMClient,
    ) -> None:
        failing_retriever = _FakeRetriever(raise_exc=RuntimeError("qdrant is down"))
        pipeline = BaselineRAG(
            llm=fake_llm,
            retriever=failing_retriever,
            settings=object(),
        )
        with pytest.raises(RuntimeError, match="qdrant is down"):
            pipeline.answer("any question")

        assert fake_llm.calls == []
