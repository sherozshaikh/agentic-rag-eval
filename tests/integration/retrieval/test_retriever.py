from __future__ import annotations

import uuid

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("fastembed")

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from agentic_rag_eval.config import get_settings
from agentic_rag_eval.retrieval.embeddings import DenseEmbedder, SparseEmbedder
from agentic_rag_eval.retrieval.qdrant_client import (
    collection_exists,
    get_qdrant_client,
)
from agentic_rag_eval.retrieval.retriever import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    Retriever,
)
from agentic_rag_eval.schemas import RetrievalStrategy

pytestmark = pytest.mark.integration

_CORPUS = [
    {
        "passage_id": "doc_einstein",
        "title": "Albert Einstein",
        "text": (
            "Albert Einstein was a German-born theoretical physicist, widely held "
            "to be one of the greatest and most influential scientists of all time."
        ),
    },
    {
        "passage_id": "doc_photosynthesis",
        "title": "Photosynthesis",
        "text": (
            "Photosynthesis is a biological process by which plants and some other "
            "organisms convert light energy into chemical energy stored in glucose."
        ),
    },
    {
        "passage_id": "doc_python",
        "title": "Python (programming language)",
        "text": (
            "Python is a high-level, general-purpose programming language known for "
            "its readability and broad ecosystem of scientific libraries."
        ),
    },
    {
        "passage_id": "doc_everest",
        "title": "Mount Everest",
        "text": (
            "Mount Everest is Earth's highest mountain above sea level, located in "
            "the Mahalangur Himal subrange of the Himalayas."
        ),
    },
]


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def qdrant_client(settings):
    """Create a client and skip the whole module if Qdrant is unreachable."""
    try:
        client = get_qdrant_client(settings)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"qdrant server not reachable: {exc}")
    return client


@pytest.fixture(scope="module")
def test_collection(qdrant_client, settings):
    """Create a throwaway collection, index the corpus, yield its name, clean up."""
    collection = f"test_retriever_{uuid.uuid4().hex[:8]}"
    dense = DenseEmbedder(model_name=settings.dense_model)
    sparse = SparseEmbedder(model_name=settings.sparse_model)

    texts = [d["text"] for d in _CORPUS]
    dense_vectors = dense.embed_passages(texts)
    sparse_vectors = sparse.embed_passages(texts)

    vector_size = len(dense_vectors[0])

    qdrant_client.recreate_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    points = []
    for i, (doc, dvec, svec) in enumerate(zip(_CORPUS, dense_vectors, sparse_vectors, strict=True)):
        points.append(
            models.PointStruct(
                id=i,
                vector={
                    DENSE_VECTOR_NAME: dvec,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=svec.indices, values=svec.values
                    ),
                },
                payload=doc,
            )
        )
    qdrant_client.upsert(collection_name=collection, points=points, wait=True)

    yield collection

    try:
        qdrant_client.delete_collection(collection_name=collection)
    except UnexpectedResponse:
        pass


@pytest.fixture()
def retriever(qdrant_client, test_collection, settings) -> Retriever:
    return Retriever(
        client=qdrant_client,
        collection_name=test_collection,
        settings=settings,
    )


class TestCollectionHelpers:
    def test_collection_exists_true(self, qdrant_client, test_collection) -> None:
        assert collection_exists(qdrant_client, test_collection) is True

    def test_collection_exists_false(self, qdrant_client) -> None:
        assert collection_exists(qdrant_client, "definitely_not_a_collection") is False


class TestDenseRetrieval:
    def test_returns_topical_match(self, retriever: Retriever) -> None:
        result = retriever.retrieve(
            "theory of relativity physicist",
            strategy=RetrievalStrategy.DENSE,
            top_k=2,
        )
        assert result.strategy == RetrievalStrategy.DENSE
        assert result.latency_ms > 0
        assert len(result.passages) >= 1
        assert result.passages[0].passage_id == "doc_einstein"
        assert result.passages[0].source_strategy == RetrievalStrategy.DENSE


class TestSparseRetrieval:
    def test_returns_lexical_match(self, retriever: Retriever) -> None:
        result = retriever.retrieve(
            "Mount Everest Himalayas",
            strategy=RetrievalStrategy.SPARSE,
            top_k=2,
        )
        assert result.strategy == RetrievalStrategy.SPARSE
        assert len(result.passages) >= 1
        assert any(p.passage_id == "doc_everest" for p in result.passages)
        assert result.passages[0].source_strategy == RetrievalStrategy.SPARSE


class TestHybridRetrieval:
    def test_rrf_fusion_returns_results(self, retriever: Retriever) -> None:
        result = retriever.retrieve(
            "Python programming language",
            strategy=RetrievalStrategy.HYBRID,
            top_k=3,
        )
        assert result.strategy == RetrievalStrategy.HYBRID
        assert len(result.passages) >= 1
        assert result.passages[0].passage_id == "doc_python"
        assert result.passages[0].source_strategy == RetrievalStrategy.HYBRID
        assert result.latency_ms > 0


class TestRetrieverValidation:
    def test_empty_query_raises(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError):
            retriever.retrieve("", strategy=RetrievalStrategy.DENSE, top_k=5)

    def test_non_positive_top_k_raises(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError):
            retriever.retrieve("x", strategy=RetrievalStrategy.DENSE, top_k=0)
