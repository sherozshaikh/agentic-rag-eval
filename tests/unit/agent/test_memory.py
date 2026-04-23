from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_rag_eval.agent.memory import MemoryStore
from agentic_rag_eval.config import Settings
from agentic_rag_eval.schemas import Passage, QueryResponse, QueryType


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(mem0_storage_path=tmp_path / "mem0")


def test_memory_store_disabled_when_mem0_missing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mem0" or name.startswith("mem0."):
            raise ImportError("simulated missing mem0")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    store = MemoryStore(settings=settings)
    assert store.enabled is False

    store.add("user-1", "hello", {"k": "v"})
    assert store.search("user-1", "query") == []


def test_memory_store_add_delegates_to_client(settings: Settings) -> None:
    client = MagicMock()
    store = MemoryStore(settings=settings, client=client)

    store.add("user-42", "some content", {"tag": "x"})
    client.add.assert_called_once()
    kwargs = client.add.call_args.kwargs
    assert kwargs["user_id"] == "user-42"
    assert kwargs["messages"] == "some content"
    assert kwargs["metadata"] == {"tag": "x"}


def test_memory_store_search_normalizes_list(settings: Settings) -> None:
    client = MagicMock()
    client.search.return_value = [
        {"memory": "fact 1"},
        {"text": "fact 2"},
        "not a dict",
    ]
    store = MemoryStore(settings=settings, client=client)

    results = store.search("user-1", "q", limit=5)
    assert len(results) == 2
    assert results[0]["memory"] == "fact 1"


def test_memory_store_search_normalizes_dict_wrapper(settings: Settings) -> None:
    client = MagicMock()
    client.search.return_value = {"results": [{"memory": "m1"}, {"memory": "m2"}]}
    store = MemoryStore(settings=settings, client=client)

    results = store.search("user-1", "q")
    assert len(results) == 2


def test_memory_store_search_swallows_exceptions(settings: Settings) -> None:
    client = MagicMock()
    client.search.side_effect = RuntimeError("mem0 down")
    store = MemoryStore(settings=settings, client=client)

    assert store.search("user-1", "q") == []


def test_memory_store_add_swallows_exceptions(settings: Settings) -> None:
    client = MagicMock()
    client.add.side_effect = RuntimeError("mem0 down")
    store = MemoryStore(settings=settings, client=client)
    store.add("user-1", "content")


def test_memory_store_add_from_response(settings: Settings) -> None:
    client = MagicMock()
    store = MemoryStore(settings=settings, client=client)

    response = QueryResponse(
        answer="42",
        confidence="high",
        query_type=QueryType.SINGLE_HOP,
        evidence=[
            Passage(passage_id="p1", title="Life, The Universe", text="..."),
            Passage(passage_id="p2", title=None, text="..."),
        ],
        trace_id="trace-abc",
    )

    store.add_from_response("user-9", "What is the answer?", response)
    client.add.assert_called_once()
    kwargs = client.add.call_args.kwargs
    assert "Q: What is the answer?" in kwargs["messages"]
    assert "A: 42" in kwargs["messages"]
    meta = kwargs["metadata"]
    assert meta["query_type"] == "single_hop"
    assert meta["trace_id"] == "trace-abc"
    assert "Life, The Universe" in meta["evidence_titles"]


def test_memory_store_noop_when_disabled(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mem0":
            raise ImportError()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    store = MemoryStore(settings=settings)
    response = QueryResponse(answer="x")
    store.add_from_response("user-1", "q", response)
