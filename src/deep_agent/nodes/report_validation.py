import re

from deep_agent.models.state import InvestigationState


def validate_report_node(state: InvestigationState) -> dict:
    report = state.get("final_report")
    if report is None:
        return {
            "report_validation_errors": ["No final report was generated."],
            "insufficient_evidence": True,
            "current_stage": "report_invalid",
        }

    valid_ids = {item.id for item in state.get("evidence", [])}
    errors: list[str] = []
    understanding = state.get("query_understanding")
    is_incident = bool(
        understanding is None
        or understanding.intent == "incident_investigation"
    )
    report.supporting_evidence_ids = [
        item for item in report.supporting_evidence_ids if item in valid_ids
    ]
    report.contradicting_evidence_ids = [
        item for item in report.contradicting_evidence_ids if item in valid_ids
    ]
    if report.root_cause and not report.supporting_evidence_ids:
        errors.append("Root cause had no valid supporting evidence IDs.")
        report.root_cause = None
        report.confidence = 0.0
        report.verification_status = "inconclusive"
        report.investigation_status = "insufficient_evidence"
    if not is_incident and report.root_cause:
        errors.append(
            "Root cause was removed because the request was not an incident."
        )
        report.root_cause = None
        report.suggested_actions = []
        report.recommended_fix = []
        if report.verification_status == "verified":
            report.investigation_status = "resolved"

    for field in ("customer_response", "engineering_note"):
        value = getattr(report, field)
        setattr(report, field, _redact(value))

    insufficient = report.root_cause is None and report.verification_status != "verified"
    if insufficient:
        report.investigation_status = "insufficient_evidence"
    return {
        "final_report": report,
        "report_validation_errors": errors,
        "insufficient_evidence": insufficient,
        "current_stage": "completed",
    }


def _redact(value: str) -> str:
    patterns = (
        r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|oracle)://[^\s]+",
        r"\b[a-f0-9]{24}\b",
        r"\b[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-"
        r"[89ab][a-f0-9]{3}-[a-f0-9]{12}\b",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b(?:password|secret|token|api[_ -]?key|credential)"
        r"\s*[:=]\s*[^\s,;]+",
    )
    result = value
    for pattern in patterns:
        result = re.sub(pattern, "[redacted]", result, flags=re.IGNORECASE)
    return result
