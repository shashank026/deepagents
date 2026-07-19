from deep_agent.models.state import InvestigationState

MIN_ROOT_CAUSE_CONFIDENCE = 0.70


def validate_root_cause_node(state: InvestigationState) -> dict:
    analysis = state.get("root_cause_analysis")
    investigation = state.get("investigation")
    if analysis is None or investigation is None:
        return {"failure_reason": "Root-cause analysis was not generated.",
                "current_stage": "inconclusive"}
    evidence_ids = {item.id for item in state.get("evidence", [])}
    hypothesis_ids = {item.id for item in investigation.hypotheses}
    analysis.supporting_evidence_ids = [x for x in analysis.supporting_evidence_ids if x in evidence_ids]
    analysis.contradicting_evidence_ids = [x for x in analysis.contradicting_evidence_ids if x in evidence_ids]
    established = bool(
        analysis.is_established and analysis.root_cause
        and analysis.confidence >= MIN_ROOT_CAUSE_CONFIDENCE
        and analysis.selected_hypothesis_id in hypothesis_ids
        and analysis.supporting_evidence_ids
    )
    if not established:
        analysis.is_established = False
        analysis.root_cause = None
        if not analysis.missing_information:
            analysis.missing_information.append(
                "The available evidence does not establish a sufficiently supported causal explanation."
            )
    return {"root_cause_analysis": analysis,
            "failure_reason": None if established else "Inconclusive root cause.",
            "current_stage": "final_report" if established else "inconclusive"}
