import json
import re

from deep_agent.models.evidence import Evidence, EvidenceType
from deep_agent.models.state import InvestigationState


def final_answer_evidence(state: InvestigationState) -> Evidence | None:
    candidates = [
        item
        for item in state.get("evidence", [])
        if (
            item.evidence_type == EvidenceType.DATABASE_QUERY
            and item.content.get("evidence_role") == "final_answer"
            and item.content.get("error") is None
            and isinstance(item.content.get("rows"), list)
            and item.content["rows"]
        )
    ]
    if not candidates:
        return None
    evidence = candidates[-1]
    return evidence if _satisfies_request(state.get("user_query", ""), evidence) else None


def _satisfies_request(query: str, evidence: Evidence) -> bool:
    lowered = query.lower()
    rows = evidence.content["rows"]
    operation = json.dumps(
        {
            "query": evidence.content.get("query"),
            "pipeline": evidence.content.get("pipeline"),
            "sort": evidence.content.get("sort"),
            "limit": evidence.content.get("limit"),
        },
        default=str,
    ).lower()
    superlative = any(term in lowered for term in (
        "highest", "lowest", "maximum", "minimum", " max ", " min ",
    ))
    if superlative:
        if not any(term in operation for term in ("$sort", "order by", "\"sort\"")):
            return False
        if len(rows) != 1:
            return False

    record = rows[0]
    normalized_fields = {
        _normalize(key): value
        for key, value in record.items()
    }
    query_tokens = {
        _singular(token)
        for token in re.findall(r"[a-z0-9]+", lowered)
        if len(token) > 2
    }

    # Explicitly requested output attributes must be present, regardless of
    # business domain. This list describes field roles, not entity names.
    attribute_roles = {
        "name", "email", "phone", "status", "balance", "amount", "count",
        "total", "currency", "latency", "duration", "date", "time", "score",
        "rate", "percentage", "version", "title", "label",
    }
    requested_roles = query_tokens & attribute_roles
    if requested_roles and not all(
        any(role in field for field in normalized_fields)
        for role in requested_roles
    ):
        return False

    if superlative:
        numeric_fields = [
            field
            for field, value in normalized_fields.items()
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and field not in {"id", "v", "version"}
                and not field.endswith("id")
            )
        ]
        metric_is_relevant = any(
            any(token in field or field in token for token in query_tokens)
            or any(role in field for role in (
                "count", "total", "amount", "balance", "score", "rate",
                "latency", "duration", "value", "percentage",
            ))
            for field in numeric_fields
        )
        if not numeric_fields or not metric_is_relevant:
            return False

    # "Who/which entity" and "identify the entity with..." requests require a
    # human-meaningful identity value. An opaque ID alone is not an answer.
    identity_question = bool(
        re.match(r"^\s*(who|which)\b", lowered)
        or re.match(
            r"^\s*(identify|find|get|give|show)\s+(me\s+)?(the\s+)?.+\s+"
            r"(with|that|who)\b",
            lowered,
        )
    )
    if identity_question and not _has_human_identity(normalized_fields):
        return False
    return True


def _has_human_identity(fields: dict[str, object]) -> bool:
    preferred = ("name", "title", "label", "email", "username", "display")
    for field, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if any(term in field for term in preferred):
            return True
        if (
            field not in {"id", "status", "type", "currency"}
            and not field.endswith("id")
            and not field.endswith("at")
        ):
            return True
    return False


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _singular(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 3:
        return value[:-1]
    return value
