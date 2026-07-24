from deep_agent.models.query import EvidenceSource, EvidenceSourcePlan
from deep_agent.models.state import InvestigationState


def plan_evidence_sources_node(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    query = state["user_query"].lower()
    sources: list[EvidenceSource] = []
    reasons: dict[EvidenceSource, str] = {}

    semantic_terms = {"success", "successful", "failed", "failure", "pending", "status"}
    code_terms = {"mapping", "enum", "implementation", "code", "logic", "configured", "configuration"}
    explicit_code = any(term in query for term in code_terms | {"source code", "where is"})
    explicit_logs = any(term in query for term in {"logs", "log entries", "search logs", "show logs"})
    incident = understanding.intent == "incident_investigation"
    has_concrete_identifiers = any(entity.value is not None for entity in understanding.entities)

    # Explicit source lookups can be source-only; business data and incidents
    # retain database evidence as the factual system-of-record baseline.
    if not ((explicit_code or explicit_logs) and not incident
            and not has_concrete_identifiers):
        sources.append(EvidenceSource.DATABASE)
        reasons[EvidenceSource.DATABASE] = (
            "Resolve entities, schemas, stored values, and records."
        )

    if (understanding.intent == "incident_investigation"
            or semantic_terms.intersection(understanding.business_terms)
            or explicit_code):
        sources.append(EvidenceSource.CODEBASE)
        reasons[EvidenceSource.CODEBASE] = (
            "Verify business meanings, encoded statuses, mappings, or application logic."
        )

    log_terms = {"error", "exception", "timeout", "incident", "timeline", "logs", "runtime"}
    if incident or explicit_logs or any(term in query for term in log_terms):
        sources.append(EvidenceSource.LOGS)
        reasons[EvidenceSource.LOGS] = "Verify runtime events, failures, and event chronology."

    # Preserve order while removing duplicates introduced by overlapping rules.
    sources = list(dict.fromkeys(sources))
    plan = EvidenceSourcePlan(sources=sources, reasons=reasons)
    return {"evidence_source_plan": plan, "current_stage": "evidence_collection"}
