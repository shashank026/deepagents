from deep_agent.models.state import InvestigationState

MAX_COLLECTION_ATTEMPTS = 3


def validate_evidence_node(state: InvestigationState) -> dict:
    valid = [
        item for item in state.get("evidence", [])
        if item.id and item.source and item.summary and isinstance(item.content, dict)
    ]
    unique = {item.id: item for item in valid}
    if not unique:
        errors = state.get("evidence_collection_errors", [])
        reason = "No valid evidence was collected."
        if errors:
            reason = f"{reason} Evidence collection error: {errors[-1]}"
        return {"evidence": [], "failure_reason": reason,
                "current_stage": "evidence_insufficient"}
    return {"evidence": list(unique.values()), "failure_reason": None,
            "current_stage": "investigation"}


def route_after_evidence_validation(state: InvestigationState) -> str:
    if state.get("evidence"):
        return "investigate"
    if state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS:
        return "collect_more_evidence"
    return "build_inconclusive_report"
