from deep_agent.models.state import InvestigationState
from deep_agent.models.evidence import EvidenceType

MAX_COLLECTION_ATTEMPTS = 3


def validate_evidence_node(state: InvestigationState) -> dict:
    valid = [
        item for item in state.get("evidence", [])
        if item.id and item.source and item.summary and isinstance(item.content, dict)
    ]
    unique = {item.id: item for item in valid}
    substantive = [
        item for item in unique.values()
        if item.evidence_type != EvidenceType.USER_INPUT
    ]
    if not unique or (
        state.get("evidence_source_plan")
        and state["evidence_source_plan"].sources
        and not substantive
    ):
        errors = state.get("evidence_collection_errors", [])
        reason = "No valid evidence was collected."
        if errors:
            reason = f"{reason} Evidence collection error: {errors[-1]}"
        return {"evidence": list(unique.values()), "failure_reason": reason,
                "current_stage": "evidence_insufficient"}
    evidence = list(unique.values())
    plan = [
        _assess_step(step, evidence)
        for step in state.get("investigation_plan", [])
    ]
    return {
        "evidence": evidence,
        "investigation_plan": plan,
        "failure_reason": None,
        "current_stage": "investigation",
    }


def route_after_evidence_validation(state: InvestigationState) -> str:
    if state.get("evidence"):
        return "investigate"
    if state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS:
        return "collect_more_evidence"
    return "build_inconclusive_report"


def _assess_step(step, evidence):
    matched = []
    blocked = False
    for item in evidence:
        if step.stage == "codebase" and item.evidence_type == EvidenceType.CODE_REFERENCE:
            content = item.content
            result = content.get("result")
            if (
                content.get("match_count", 0) > 0
                or content.get("snippets")
                or (isinstance(result, dict) and result.get("content"))
                or content.get("matches")
            ):
                matched.append(item.id)
        elif step.stage == "schema" and item.evidence_type == EvidenceType.DATABASE_SCHEMA:
            matched.append(item.id)
        elif step.stage == "database" and item.evidence_type in {
            EvidenceType.DATABASE_QUERY, EvidenceType.DATABASE_RECORD,
        }:
            if not item.content.get("error"):
                matched.append(item.id)
        elif step.stage == "logs" and item.evidence_type in {
            EvidenceType.LOG_ENTRY, EvidenceType.CONFIGURATION,
        }:
            if item.content.get("unavailable") or item.content.get("error"):
                blocked = True
            else:
                matched.append(item.id)
        elif (
            step.stage == "web"
            and item.content.get("external_context_only") is True
        ):
            if item.content.get("unavailable") or item.content.get("error"):
                blocked = True
            elif (
                item.content.get("citations")
                or item.content.get("url")
            ):
                matched.append(item.id)
    if step.stage == "validation":
        return step
    if matched:
        return step.model_copy(update={
            "status": "completed",
            "evidence_ids": list(dict.fromkeys(matched)),
        })
    if blocked:
        return step.model_copy(update={"status": "blocked"})
    return step.model_copy(update={"status": "pending"})
