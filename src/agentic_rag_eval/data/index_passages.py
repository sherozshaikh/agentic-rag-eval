from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import pandas as pd

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_BATCH_SIZE = 128


def _to_qdrant_id(s: str) -> int:
    """Convert arbitrary string to deterministic unsigned int for Qdrant."""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)


def _load_passages(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Passages parquet not found: {path}. "
            "Run `make seed` or `python -m agentic_rag_eval.data.seed` first."
        )
    df = pd.read_parquet(path)
    required = {"passage_id", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Passages parquet missing columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError(f"Passages parquet is empty: {path}")
    return df


def _build_qdrant_client(settings: Settings) -> QdrantClient:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required. `pip install qdrant-client[fastembed]`."
        ) from exc

    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_use_grpc,
            timeout=30.0,
            check_compatibility=False,
        )
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Qdrant at {settings.qdrant_host}:{settings.qdrant_port}. "
            "Is the container running? Try `docker compose up qdrant`."
        ) from exc
    return client


def _ensure_collection(
    client: QdrantClient,
    collection: str,
    dense_dim: int,
) -> bool:
    """Create the collection if missing; return True when newly created."""
    from qdrant_client.http import models as qmodels

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        vectors_cfg = getattr(info.config.params, "vectors", None) or {}
        sparse_cfg = getattr(info.config.params, "sparse_vectors", None) or {}
        has_dense = isinstance(vectors_cfg, dict) and DENSE_VECTOR_NAME in vectors_cfg
        has_sparse = isinstance(sparse_cfg, dict) and SPARSE_VECTOR_NAME in sparse_cfg
        if has_dense and has_sparse:
            logger.info(
                "Qdrant collection already exists with matching config — skipping create",
                extra={"collection": collection},
            )
            return False
        raise RuntimeError(
            f"Collection {collection!r} exists but does not have the required "
            f"named vectors {DENSE_VECTOR_NAME!r}/{SPARSE_VECTOR_NAME!r}. "
            "Drop it and re-run, or change the collection name in your config."
        )

    logger.info(
        "Creating Qdrant collection",
        extra={"collection": collection, "dense_dim": dense_dim},
    )
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=dense_dim,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False),
                modifier=qmodels.Modifier.IDF,
            ),
        },
    )
    return True


def _build_embedders(settings: Settings) -> tuple[Any, Any, int]:
    try:
        from fastembed import SparseTextEmbedding, TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required. `pip install fastembed`.") from exc

    dense = TextEmbedding(model_name=settings.dense_model)
    sparse = SparseTextEmbedding(model_name=settings.sparse_model)

    probe = next(iter(dense.embed(["probe"])))
    dense_dim = len(probe)
    return dense, sparse, dense_dim


def _batched(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def index_passages(
    passages_path: Path,
    settings: Settings | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
) -> int:
    """Index a passages parquet into Qdrant and return the upsert count."""
    settings = settings or get_settings()
    df = _load_passages(passages_path)
    logger.info(
        "Loaded passages for indexing",
        extra={"num_passages": len(df), "path": str(passages_path)},
    )

    client = _build_qdrant_client(settings)

    from qdrant_client.http import models as qmodels
    from tqdm import tqdm

    dense_embedder, sparse_embedder, dense_dim = _build_embedders(settings)

    if recreate:
        try:
            client.delete_collection(settings.qdrant_collection)
            logger.info(
                "Dropped existing collection (recreate=True)",
                extra={"collection": settings.qdrant_collection},
            )
        except Exception:
            pass

    created = _ensure_collection(client, settings.qdrant_collection, dense_dim=dense_dim)
    if not created and not recreate:
        return 0

    total_upserted = 0
    with tqdm(total=len(df), desc="Indexing passages", unit="psg") as pbar:
        for batch in _batched(df, batch_size):
            texts: list[str] = batch["text"].astype(str).tolist()
            titles: list[str] = batch["title"].astype(str).tolist()
            ids: list[str] = batch["passage_id"].astype(str).tolist()

            dense_vecs = list(dense_embedder.embed(texts))
            sparse_vecs = list(sparse_embedder.embed(texts))

            points: list[qmodels.PointStruct] = []

            for pid, title, text, dense_v, sparse_v in zip(
                ids, titles, texts, dense_vecs, sparse_vecs, strict=True
            ):
                qdrant_id = _to_qdrant_id(pid)

                logger.debug(
                    "Mapping passage_id → qdrant_id",
                    extra={"pid": pid, "qid": qdrant_id},
                )

                point = qmodels.PointStruct(
                    id=qdrant_id,
                    vector={
                        DENSE_VECTOR_NAME: list(map(float, dense_v)),
                        SPARSE_VECTOR_NAME: qmodels.SparseVector(
                            indices=list(map(int, sparse_v.indices)),
                            values=list(map(float, sparse_v.values)),
                        ),
                    },
                    payload={"passage_id": pid, "title": title, "text": text},
                )
                points.append(point)

            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
                wait=True,
            )
            total_upserted += len(points)
            pbar.update(len(points))

    logger.info(
        "Finished indexing passages",
        extra={
            "collection": settings.qdrant_collection,
            "num_upserted": total_upserted,
        },
    )
    return total_upserted


@click.command()
@click.option(
    "--passages",
    "passages_path",
    type=click.Path(path_type=Path),
    default=Path("data/processed/passages.parquet"),
    show_default=True,
    help="Path to the passages parquet written by the seed pipeline.",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
)
@click.option(
    "--recreate",
    is_flag=True,
    default=False,
    help="Drop and recreate the collection before indexing.",
)
def main(passages_path: Path, batch_size: int, recreate: bool) -> None:
    """Index HotpotQA passages into Qdrant."""
    try:
        n = index_passages(
            passages_path=passages_path,
            batch_size=batch_size,
            recreate=recreate,
        )
    except RuntimeError as exc:
        logger.error("Indexing failed", extra={"error": str(exc)})
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Upserted {n} passages.")


if __name__ == "__main__":
    main()
