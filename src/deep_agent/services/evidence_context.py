import json
from typing import Any, Iterable

from deep_agent.models.evidence import Evidence, EvidenceType


def compact_evidence(
    evidence: Iterable[Evidence],
    *,
    max_items: int = 24,
    referenced_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a bounded, deduplicated evidence payload for model evaluation."""
    unique: dict[str, Evidence] = {}
    for item in evidence:
        signature = json.dumps(
            {
                "type": item.evidence_type.value,
                "source": item.source,
                "summary": item.summary,
                "operation": _operation_identity(item.content),
            },
            sort_keys=True,
            default=str,
        )
        unique[signature] = item

    items = list(unique.values())
    referenced = referenced_ids or set()
    items.sort(
        key=lambda item: (
            item.id not in referenced,
            _priority(item),
            item.collected_at or "",
        )
    )
    selected = items[:max_items]
    return [
        {
            "id": item.id,
            "evidence_type": item.evidence_type.value,
            "source": item.source,
            "summary": item.summary,
            "reliability": item.reliability.value,
            "content": _bounded(item.content),
        }
        for item in selected
    ]


def _operation_identity(content: dict[str, Any]) -> dict[str, Any]:
    return {
        key: content.get(key)
        for key in (
            "query", "collection", "filter", "projection", "sort", "pipeline",
            "path", "sha", "ref", "error", "evidence_role",
        )
        if key in content
    }


def _priority(item: Evidence) -> int:
    return {
        EvidenceType.USER_INPUT: 0,
        EvidenceType.LOG_ENTRY: 1,
        EvidenceType.CODE_REFERENCE: 2,
        EvidenceType.CONFIGURATION: 3,
        EvidenceType.API_RESPONSE: 4,
        EvidenceType.DATABASE_QUERY: 5,
        EvidenceType.DATABASE_RECORD: 6,
        EvidenceType.DATABASE_SCHEMA: 7,
    }.get(item.evidence_type, 8)


def _bounded(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:24]:
            normalized = str(key).lower()
            if any(term in normalized for term in (
                "password", "secret", "token", "credential",
            )):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = _bounded(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth + 1) for item in list(value)[:10]]
    if isinstance(value, str):
        return value[:5000]
    return value
