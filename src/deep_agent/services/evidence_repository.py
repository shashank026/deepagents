import asyncio
import os
from collections import defaultdict
from contextvars import ContextVar, Token

from deep_agent.models.evidence import Evidence


_active_investigation: ContextVar[str | None] = ContextVar(
    "active_investigation", default=None
)


class EvidenceRepository:
    """Process-local evidence store used by tool wrappers.

    Replace this implementation with durable storage in multi-process production
    deployments; the interface deliberately keeps workflow code independent of it.
    """

    def __init__(self, max_items_per_investigation: int | None = None) -> None:
        self._items: dict[str, list[Evidence]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._max_items = max_items_per_investigation or int(
            os.getenv("MAX_EVIDENCE_ITEMS_PER_INVESTIGATION", "500")
        )

    async def save(self, investigation_id: str, evidence: Evidence) -> None:
        async with self._lock:
            items = self._items[investigation_id]
            if any(item.id == evidence.id for item in items):
                return
            if len(items) >= self._max_items:
                raise RuntimeError(
                    "Investigation evidence storage limit was reached"
                )
            items.append(evidence)

    async def list_by_investigation(self, investigation_id: str) -> list[Evidence]:
        async with self._lock:
            return list(self._items.get(investigation_id, []))

    async def clear(self, investigation_id: str) -> None:
        async with self._lock:
            self._items.pop(investigation_id, None)


evidence_repository = EvidenceRepository()


def bind_investigation(investigation_id: str) -> Token:
    return _active_investigation.set(investigation_id)


def reset_investigation(token: Token) -> None:
    _active_investigation.reset(token)


def active_investigation_id() -> str:
    investigation_id = _active_investigation.get()
    if not investigation_id:
        raise RuntimeError("Evidence tool called outside an investigation context")
    return investigation_id
