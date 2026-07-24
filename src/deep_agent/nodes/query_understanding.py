import re

from deep_agent.models.query import BusinessEntity, QueryUnderstanding
from deep_agent.models.state import InvestigationState


ENTITY_ALIASES = {
    "booking": "booking",
    "bookings": "booking",
    "bookingid": "booking_id",
    "booking_id": "booking_id",
    "user": "user",
    "userid": "user_id",
    "user_id": "user_id",
    "customer": "customer",
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "organization": "organization",
    "organisation": "organization",
    "organizationid": "organization_id",
    "organisationid": "organization_id",
    "wallet": "wallet",
    "wallets": "wallet",
    "payment": "payment",
    "payments": "payment",
    "paymentid": "payment_id",
    "payment_id": "payment_id",
    "order": "order",
    "orders": "order",
    "orderid": "order_id",
    "order_id": "order_id",
}

BUSINESS_TERMS = {
    "success", "successful", "failed", "failure", "pending", "status",
    "refund", "refunded", "cancelled", "canceled", "timeout", "error",
}


def _intent(query: str) -> str:
    lowered = query.strip().lower()
    if re.match(
        r"^(give|show|list|get|return|fetch|find|identify|which|what|who)\b",
        lowered,
    ):
        return "data_retrieval"
    if any(term in lowered for term in ("why", "root cause", "incident", "error", "failed")):
        return "incident_investigation"
    return "analysis"


def extract_business_entities_node(state: InvestigationState) -> dict:
    query = state["user_query"]
    lowered = query.lower()
    entities: list[BusinessEntity] = []
    seen: set[tuple[str, str | None]] = set()

    # Extract explicit identifiers: bookingid=123, user id ABC, payment_id: X.
    id_pattern = re.compile(
        r"\b(booking|user|customer|payment|order)[_\s-]*id\s*(?:=|:|is)?\s*"
        r"([a-z0-9][a-z0-9_-]*)\b",
        re.IGNORECASE,
    )
    for match in id_pattern.finditer(query):
        entity_type = f"{match.group(1).lower()}_id"
        key = (entity_type, match.group(2))
        if key not in seen:
            seen.add(key)
            entities.append(BusinessEntity(
                entity_type=entity_type, value=match.group(2),
                source_text=match.group(0), confidence=1.0,
            ))

    # Also retain mentioned domain objects even when no concrete ID is supplied.
    for token, entity_type in ENTITY_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            key = (entity_type, None)
            if key not in seen and not any(
                item.entity_type == entity_type and item.value is not None
                for item in entities
            ):
                seen.add(key)
                entities.append(BusinessEntity(
                    entity_type=entity_type, source_text=token, confidence=0.9
                ))

    terms = sorted(term for term in BUSINESS_TERMS if re.search(rf"\b{term}\b", lowered))
    understanding = QueryUnderstanding(
        intent=_intent(query), entities=entities, business_terms=terms
    )
    return {
        "query_understanding": understanding,
        "extracted_entities": {
            item.entity_type: item.value for item in entities if item.value is not None
        },
        "current_stage": "source_planning",
    }
