import re

from deep_agent.models.query import (
    BusinessEntity, QueryConstraint, QueryUnderstanding,
)
from deep_agent.models.state import InvestigationState


BUSINESS_TERMS = {
    "success", "successful", "failed", "failure", "pending", "status",
    "refund", "refunded", "cancelled", "canceled", "timeout", "error",
}


def _intent(query: str) -> str:
    lowered = _authoritative_request_text(query)
    if any(term in lowered for term in (
        "why", "root cause", "incident", "error", "failed", "failure",
        "not able", "unable", "doesn't work", "does not work",
    )):
        return "incident_investigation"
    if any(term in lowered for term in (
        "how", "explain", "what happens", "behavior", "behaviour",
        "flow", "works", "logic",
    )):
        return "explanation"
    if any(term in lowered for term in (
        "documentation", "policy", "specification", "availability",
        "available",
    )):
        return "informational"
    if re.match(r"^what\s+(?:is|are)\b", lowered) and any(
        term in lowered for term in (
            "last ", "latest", "newest", "oldest", "most recent",
            "highest", "lowest", "maximum", "minimum",
        )
    ):
        return "data_retrieval"
    if re.match(
        r"^(what\s+(?:is|are)|tell\s+me\s+about|provide\s+information)\b",
        lowered,
    ):
        return "informational"
    if re.match(
        r"^(give|show|list|get|return|fetch|find|identify|which|what|who)\b",
        lowered,
    ):
        return "data_retrieval"
    return "analysis"


def _authoritative_request_text(query: str) -> str:
    """Prefer the latest explicit email-thread question over subject metadata."""
    lowered = query.strip().lower()
    message_questions: list[str] = []
    for match in re.finditer(
        r"(?:^|\n)message:\s*(.*?)(?=\n\n---|\Z)",
        lowered,
        flags=re.DOTALL,
    ):
        question = re.sub(r"[*_`]+", "", match.group(1)).strip()
        if "?" in question:
            message_questions.append(question)
    return message_questions[-1] if message_questions else lowered


def extract_business_entities_node(state: InvestigationState) -> dict:
    query = state["user_query"]
    lowered = query.lower()
    entities: list[BusinessEntity] = []
    constraints: list[QueryConstraint] = []
    seen: set[tuple[str, str | None]] = set()

    # Extract arbitrary domain identifiers without a fixed business vocabulary.
    id_pattern = re.compile(
        r"\b((?!did\b)[a-z][a-z0-9_.-]*[_-]?id|"
        r"[a-z][a-z0-9_.-]*\s+id)"
        r"\s*(?:=|:|is)?\s*"
        r"([a-z0-9][a-z0-9_-]*)\b",
        re.IGNORECASE,
    )
    for match in id_pattern.finditer(query):
        entity_type = re.sub(
            r"[\s.-]+", "_", match.group(1).lower()
        )
        if entity_type.endswith("id") and not entity_type.endswith("_id"):
            entity_type = f"{entity_type[:-2].rstrip('_')}_id"
        key = (entity_type, match.group(2))
        if key not in seen:
            seen.add(key)
            entities.append(BusinessEntity(
                entity_type=entity_type, value=match.group(2),
                source_text=match.group(0), confidence=1.0,
            ))

    # Capture arbitrary domain filters instead of limiting understanding to a
    # fixed list of business objects: account_key=..., project: ..., sku is ...
    generic_constraint = re.compile(
        r"\b([a-z][a-z0-9_.-]{1,63})\s*(?:=|:|\bis\b)\s*"
        r"(\"[^\"]+\"|'[^']+'|[a-z0-9@._+/-]+)",
        re.IGNORECASE,
    )
    for match in generic_constraint.finditer(query):
        field_hint = match.group(1).strip().lower().replace("-", "_")
        value = match.group(2).strip().strip("\"'").rstrip(".,;")
        if not value or any(
            item.field_hint == field_hint and item.value == value
            for item in constraints
        ):
            continue
        constraints.append(QueryConstraint(
            field_hint=field_hint,
            value=value,
            source_text=match.group(0),
            confidence=0.95,
        ))

    for match in re.finditer(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        query,
        re.IGNORECASE,
    ):
        value = match.group(0).lower()
        if not any(item.value == value for item in constraints):
            constraints.append(QueryConstraint(
                field_hint="email",
                value=value,
                source_text=match.group(0),
                confidence=1.0,
            ))

    # Capture likely requested entity nouns by grammar rather than a
    # product-specific dictionary. The evidence agent still verifies their
    # meaning against this project's schema and code.
    entity_patterns = (
        r"\b(?:identify|find|which|who)\s+(?:the\s+)?([a-z][a-z0-9_-]*)",
        r"\b(?:highest|lowest|latest|oldest|largest|smallest)\s+"
        r"([a-z][a-z0-9_-]*)",
    )
    for pattern in entity_patterns:
        for match in re.finditer(pattern, lowered):
            entity_type = match.group(1).rstrip("s")
            if entity_type in {"one", "record", "result"}:
                continue
            key = (entity_type, None)
            if key not in seen:
                seen.add(key)
                entities.append(BusinessEntity(
                    entity_type=entity_type,
                    source_text=match.group(0),
                    confidence=0.65,
                ))

    terms = sorted(term for term in BUSINESS_TERMS if re.search(rf"\b{term}\b", lowered))
    understanding = QueryUnderstanding(
        intent=_intent(query),
        entities=entities,
        business_terms=terms,
        constraints=constraints,
        requested_output=query.strip(),
    )
    return {
        "query_understanding": understanding,
        "extracted_entities": {
            item.entity_type: item.value for item in entities if item.value is not None
        },
        "current_stage": "source_planning",
    }
