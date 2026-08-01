from deep_agent.models.state import InvestigationState
from deep_agent.models.evidence import EvidenceType

MIN_ROOT_CAUSE_CONFIDENCE = 0.70
_UNSAFE_REMEDIATION_MARKERS = (
    "manually create",
    "directly create",
    "manually insert",
    "directly insert",
    "manually update",
    "directly update",
    "repair the subscription record",
)
_SPECULATIVE_MARKERS = (
    "potential ",
    "possibly ",
    "may have ",
    "might have ",
    "suggests a ",
)


def _is_unsafe_remediation(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _UNSAFE_REMEDIATION_MARKERS)


def _is_speculative(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SPECULATIVE_MARKERS)


def is_causal_support(item) -> bool:
    """Reject investigation artifacts as proof of customer-facing causality."""
    if item.content.get("error") or item.content.get("unavailable") is True:
        return False
    if item.evidence_type == EvidenceType.DATABASE_QUERY:
        return (
            item.content.get("evidence_role") == "causal_validation"
            and bool(item.content.get("rows"))
        )
    return item.evidence_type in {
        EvidenceType.LOG_ENTRY,
        EvidenceType.CODE_REFERENCE,
        EvidenceType.CONFIGURATION,
        EvidenceType.API_RESPONSE,
    } and item.content.get("external_context_only") is not True


def validate_root_cause_node(state: InvestigationState) -> dict:
    analysis = state.get("root_cause_analysis")
    investigation = state.get("investigation")
    if analysis is None or investigation is None:
        return {"failure_reason": "Root-cause analysis was not generated.",
                "current_stage": "inconclusive"}
    evidence_ids = {item.id for item in state.get("evidence", [])}
    evidence_by_id = {item.id: item for item in state.get("evidence", [])}
    hypothesis_ids = {item.id for item in investigation.hypotheses}
    analysis.supporting_evidence_ids = [x for x in analysis.supporting_evidence_ids if x in evidence_ids]
    analysis.contradicting_evidence_ids = [x for x in analysis.contradicting_evidence_ids if x in evidence_ids]
    has_internal_support = any(
        is_causal_support(evidence_by_id[evidence_id])
        for evidence_id in analysis.supporting_evidence_ids
    )
    established = bool(
        analysis.is_established and analysis.root_cause
        and analysis.confidence >= MIN_ROOT_CAUSE_CONFIDENCE
        and analysis.selected_hypothesis_id in hypothesis_ids
        and analysis.supporting_evidence_ids
        and has_internal_support
    )
    # Even an established diagnosis does not authorize a generated agent to
    # prescribe direct writes to customer or billing data. Keep remediation on
    # the supported source-of-truth workflow, and do not promote speculation
    # into an evidenced contributing factor.
    analysis.recommended_fix = [
        item
        for item in analysis.recommended_fix
        if not _is_unsafe_remediation(item)
    ]
    analysis.suggested_actions = [
        item
        for item in analysis.suggested_actions
        if not _is_unsafe_remediation(item)
    ]
    analysis.contributing_factors = [
        item
        for item in analysis.contributing_factors
        if not _is_speculative(item)
    ]
    if not established:
        analysis.is_established = False
        analysis.root_cause = None
        analysis.confidence = 0.0
        analysis.supporting_evidence_ids = []
        analysis.reasoning_summary = (
            "Root cause could not be established from causal evidence. "
            "Exploratory searches and query corrections were excluded."
        )
        # Never recommend mutations or remediation when causality is not
        # established. Remaining items must describe missing evidence only.
        analysis.suggested_actions = []
        analysis.recommended_fix = []
        analysis.validation_steps = []
        analysis.contributing_factors = []
        concrete_missing = [
            item
            for item in investigation.unresolved_questions
            if item.strip()
        ]
        unavailable_sources = [
            (
                f"Unavailable evidence source: "
                f"{item.content.get('error') or item.summary}"
            )
            for item in state.get("evidence", [])
            if (
                item.content.get("unavailable") is True
                or (
                    item.evidence_type.value in {"log_entry", "configuration"}
                    and item.content.get("error")
                )
            )
        ]
        if concrete_missing:
            analysis.missing_information = list(dict.fromkeys(
                [
                    *analysis.missing_information,
                    *concrete_missing,
                    *unavailable_sources,
                ]
            ))
        elif unavailable_sources:
            analysis.missing_information = list(dict.fromkeys(
                [*analysis.missing_information, *unavailable_sources]
            ))
        if not analysis.missing_information:
            analysis.missing_information.append(
                "The available evidence does not establish a sufficiently supported causal explanation."
            )
    return {"root_cause_analysis": analysis,
            "failure_reason": None if established else "Inconclusive root cause.",
            "current_stage": "final_report" if established else "inconclusive"}
