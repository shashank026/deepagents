import re

from deep_agent.models.state import InvestigationState
from deep_agent.nodes.root_cause_validation import is_causal_support


_INCONCLUSIVE_CUSTOMER_RESPONSE = """Subject: TraceX investigation update

Hello,

We reviewed the available evidence, but it is not sufficient to confirm a reliable root cause at this time. No corrective action should be taken from exploratory searches alone.

No customer data was modified during this investigation.

If you need any additional validation or assistance, please reply to this message and our support team will be happy to help.

Regards,
TraceX L2 Support Team"""


def validate_report_node(state: InvestigationState) -> dict:
    report = state.get("final_report")
    if report is None:
        return {
            "report_validation_errors": ["No final report was generated."],
            "insufficient_evidence": True,
            "current_stage": "report_invalid",
        }

    valid_ids = {item.id for item in state.get("evidence", [])}
    evidence_by_id = {item.id: item for item in state.get("evidence", [])}
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
    analysis = state.get("root_cause_analysis")
    has_causal_support = any(
        is_causal_support(evidence_by_id[evidence_id])
        for evidence_id in report.supporting_evidence_ids
    )
    valid_analysis = bool(analysis and analysis.is_established)
    if report.root_cause and (not has_causal_support or not valid_analysis):
        errors.append(
            "Root cause did not have a validated causal analysis and causal evidence."
        )
        report.root_cause = None
        report.confidence = 0.0
        report.verification_status = "inconclusive"
        report.investigation_status = "insufficient_evidence"
        report.supporting_evidence_ids = []
        report.suggested_actions = []
        report.recommended_fix = []
        report.validation_steps = []
        report.contributing_factors = []
        report.customer_response = _INCONCLUSIVE_CUSTOMER_RESPONSE
        report.engineering_note = (
            "Root cause was withheld because the report was not backed by a "
            "validated causal analysis and causal evidence. Exploratory "
            "lookups and query corrections are not customer incident causes."
        )
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
