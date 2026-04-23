from __future__ import annotations

import os

import pytest

from agentic_rag_eval.agent import MemoryStore, ReActAgent
from agentic_rag_eval.config import get_settings
from agentic_rag_eval.llm import build_llm_client
from agentic_rag_eval.schemas import QueryResponse

pytestmark = pytest.mark.integration


REQUIRES_LIVE_STACK = not os.environ.get("AGENTIC_RAG_EVAL_RUN_INTEGRATION")


@pytest.mark.skipif(
    REQUIRES_LIVE_STACK,
    reason="Integration test — set AGENTIC_RAG_EVAL_RUN_INTEGRATION=1 with Qdrant+Ollama running",
)
def test_react_agent_end_to_end_single_hop() -> None:
    """Smoke test: the agent should produce a non-empty answer for a single-hop question."""
    from agentic_rag_eval.retrieval import AdaptiveRetriever

    settings = get_settings()
    llm = build_llm_client(settings, role="agent")
    retriever = AdaptiveRetriever()
    memory = MemoryStore(settings=settings)

    agent = ReActAgent(
        llm=llm,
        retriever=retriever,
        memory=memory,
        settings=settings,
    )

    response: QueryResponse = agent.answer(
        "Who wrote the novel 'The Great Gatsby'?",
        user_id="integration-user",
    )

    assert isinstance(response, QueryResponse)
    assert response.answer
    assert response.trace_id
    assert response.reasoning_chain
    assert response.latency_ms > 0.0


@pytest.mark.skipif(
    REQUIRES_LIVE_STACK,
    reason="Integration test — set AGENTIC_RAG_EVAL_RUN_INTEGRATION=1 with Qdrant+Ollama running",
)
def test_react_agent_end_to_end_multi_hop() -> None:
    """The agent should decompose a multi-hop question into multiple sub-questions."""
    from agentic_rag_eval.retrieval import AdaptiveRetriever

    settings = get_settings()
    llm = build_llm_client(settings, role="agent")
    retriever = AdaptiveRetriever()

    agent = ReActAgent(
        llm=llm,
        retriever=retriever,
        memory=None,
        settings=settings,
    )

    response = agent.answer(
        "What year was the director of 'Inception' born?",
    )

    assert response.answer
    assert len(response.sub_questions) >= 1
    assert response.trace_id
