import re

from deep_agent.models.evidence import EvidenceType
from deep_agent.models.report import ResultRecord, RootCauseReport
from deep_agent.models.state import InvestigationState


def _retrieval_evidence(state: InvestigationState):
    candidates = [
        item for item in state.get("evidence", [])
        if item.evidence_type == EvidenceType.DATABASE_QUERY
        and isinstance(item.content.get("rows"), list)
        and item.content["rows"]
    ]
    return max(candidates, key=lambda item: len(item.content["rows"]), default=None)


def _looks_like_retrieval(state: InvestigationState) -> bool:
    query = state.get("user_query", "").strip().lower()
    if not re.match(r"^(give|show|list|get|return|fetch)\b", query):
        return False
    return not any(term in query for term in (
        "why", "failed", "failure", "error", "incident", "root cause"
    ))


def _result_records(state: InvestigationState, final_result_only: bool = False) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    selected = _retrieval_evidence(state) if final_result_only else None
    evidence = [selected] if selected is not None else state.get("evidence", [])
    for item in evidence:
        if item.evidence_type not in {
            EvidenceType.DATABASE_RECORD, EvidenceType.DATABASE_QUERY,
            EvidenceType.API_RESPONSE,
        }:
            continue
        rows = item.content.get("rows")
        if isinstance(rows, list):
            records.extend(ResultRecord(source=item.source, record=row) for row in rows if isinstance(row, dict))
        else:
            records.append(ResultRecord(source=item.source, record=item.content))
        if len(records) >= 100:
            break
    return records[:100]


def build_final_report_node(state: InvestigationState) -> dict:
    investigation = state.get("investigation")
    analysis = state.get("root_cause_analysis")
    if investigation is None:
        reason = state.get("failure_reason") or "The investigation could not be completed."
        report = RootCauseReport(
            issue_summary=state["user_query"], missing_information=[reason],
            customer_response="We could not complete the investigation with the currently available information.",
            engineering_note="No valid investigation result was produced. Review evidence collection failures and tool traces.",
        )
    else:
        established = bool(analysis and analysis.is_established)
        retrieval_evidence = (
            _retrieval_evidence(state)
            if analysis is None and _looks_like_retrieval(state)
            else None
        )
        is_retrieval = retrieval_evidence is not None
        missing = ([] if is_retrieval else
                   (analysis.missing_information if analysis else investigation.unresolved_questions))
        if established:
            customer = f"We identified the cause: {analysis.root_cause}"
        elif is_retrieval:
            customer = "The requested records are included in the result set; no incident root cause applies."
        else:
            customer = "The available evidence does not establish a root cause."
        report = RootCauseReport(
            issue_summary=investigation.issue_summary,
            root_cause=analysis.root_cause if established else None,
            confidence=analysis.confidence if analysis else 0.0,
            supporting_evidence_ids=(analysis.supporting_evidence_ids if analysis else
                                     ([retrieval_evidence.id] if retrieval_evidence else [])),
            contradicting_evidence_ids=analysis.contradicting_evidence_ids if analysis else [],
            affected_components=investigation.affected_components,
            expected_state=investigation.expected_state,
            actual_state=investigation.actual_state,
            suggested_actions=analysis.suggested_actions if analysis else [],
            missing_information=missing,
            customer_response=customer,
            engineering_note=(analysis.reasoning_summary if analysis else
                              "Investigation completed without a root-cause analysis."),
            result_records=_result_records(state, final_result_only=is_retrieval),
        )
    return {"final_report": report, "current_stage": "completed"}
