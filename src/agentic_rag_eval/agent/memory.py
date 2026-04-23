from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import QueryResponse

logger = get_logger(__name__)


class MemoryStore:
    """Thin Mem0 wrapper; degrades to a no-op when Mem0 is unavailable."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        storage_path: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._storage_path = storage_path or self._settings.mem0_storage_path
        self._client: Any | None = client
        self._enabled = False

        if client is not None:
            self._enabled = True
            return

        self._client = self._build_client()
        self._enabled = self._client is not None

    def _build_client(self) -> Any | None:
        """Construct a Mem0 client, or return None on any failure."""
        try:
            from mem0 import Memory
        except Exception as e:
            logger.warning(
                "mem0_unavailable",
                extra={"error": str(e), "reason": "import_failed"},
            )
            return None

        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            config: dict[str, Any] = {
                "history_db_path": str(self._storage_path / "history.db"),
            }
            try:
                return Memory.from_config(config)
            except Exception:
                return Memory()
        except Exception as e:
            logger.warning(
                "mem0_init_failed",
                extra={"error": str(e), "storage_path": str(self._storage_path)},
            )
            return None

    @property
    def enabled(self) -> bool:
        """Whether Mem0 is active; when False, all calls are no-ops."""
        return self._enabled

    def add(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory entry for `user_id`; silent on failure."""
        if not self._enabled or not self._client:
            return
        try:
            self._client.add(
                messages=content,
                user_id=user_id,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning(
                "mem0_add_failed",
                extra={"error": str(e), "user_id": user_id},
            )

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return memory entries for `user_id` relevant to `query`, or empty on failure."""
        if not self._enabled or not self._client:
            return []
        try:
            raw = self._client.search(query=query, user_id=user_id, limit=limit)
        except Exception as e:
            logger.warning(
                "mem0_search_failed",
                extra={"error": str(e), "user_id": user_id, "query": query[:200]},
            )
            return []

        return self._normalize_search_results(raw)

    def add_from_response(
        self,
        user_id: str,
        question: str,
        response: QueryResponse,
    ) -> None:
        """Persist a memory entry derived from a completed `QueryResponse`."""
        if not self._enabled or not self._client:
            return

        try:
            evidence_titles = [p.title for p in response.evidence if p.title][:10]

            content = f"Q: {question}\nA: {response.answer}\nConfidence: {response.confidence}"
            metadata: dict[str, Any] = {
                "query_type": response.query_type.value,
                "confidence": response.confidence,
                "evidence_conflict": response.evidence_conflict,
                "evidence_titles": evidence_titles,
                "trace_id": response.trace_id,
            }
            self.add(user_id=user_id, content=content, metadata=metadata)
        except Exception as e:
            logger.warning(
                "mem0_add_from_response_failed",
                extra={"error": str(e), "user_id": user_id},
            )

    @staticmethod
    def _normalize_search_results(raw: Any) -> list[dict[str, Any]]:
        """Normalize Mem0 search output across supported client versions."""
        if raw is None:
            return []
        if isinstance(raw, dict):
            results = raw.get("results")
            if isinstance(results, list):
                return [r for r in results if isinstance(r, dict)]
            return []
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        return []
