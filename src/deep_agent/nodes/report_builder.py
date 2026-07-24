import re
from typing import Any

from deep_agent.models.evidence import EvidenceType
from deep_agent.models.report import ResultRecord, RootCauseReport
from deep_agent.models.state import InvestigationState
from deep_agent.services.retrieval_verification import final_answer_evidence


def _retrieval_evidence(state: InvestigationState):
    return final_answer_evidence(state)


def _looks_like_retrieval(state: InvestigationState) -> bool:
    query = state.get("user_query", "").strip().lower()
    if not re.match(
        r"^(give|show|list|get|return|fetch|identify|find|which|what|who)\b",
        query,
    ):
        return False
    return not any(term in query for term in (
        "why", "failed", "failure", "error", "incident", "root cause"
    ))


def _retrieval_is_verified(state: InvestigationState) -> bool:
    investigation = state.get("investigation")
    if (
        investigation is None
        or investigation.requires_more_evidence
        or investigation.requested_evidence
    ):
        return False
    unresolved_text = " ".join([
        investigation.actual_state,
        *investigation.unresolved_questions,
    ]).lower()
    return not any(marker in unresolved_text for marker in (
        "not explicitly defined",
        "not established",
        "cannot determine",
        "could not determine",
        "unable to determine",
        "missing relationship",
        "relationship is unknown",
        "insufficient evidence",
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


def _customer_retrieval_response(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 1:
        return _customer_email(
            "We completed the requested database review and verified "
            f"{len(rows)} matching records. The result table below contains "
            "the fields relevant to the request and reflects the data available "
            "at the time of the investigation."
        )
    record = rows[0]
    excluded_terms = {
        "address", "billing", "description", "password", "secret", "token",
        "gst", "tax", "__v", "createdat", "updatedat", "email", "phone",
        "mobile", "contact", "credential", "apikey", "auth",
    }
    priority_terms = (
        "name", "balance", "amount", "total", "count", "currency", "status",
    )
    values: list[tuple[int, str, Any]] = []
    for key, value in record.items():
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if (
            value is None
            or isinstance(value, (dict, list))
            or _is_sensitive_customer_field(normalized, excluded_terms)
        ):
            continue
        priority = next(
            (index for index, term in enumerate(priority_terms) if term in normalized),
            len(priority_terms),
        )
        values.append((priority, key, value))
    values.sort(key=lambda item: (item[0], item[1]))
    if values:
        verified_result = "Verified result:\n" + "\n".join(
            f"- {_humanize_key(key)}: {_format_customer_value(value)}"
            for _, key, value in values[:8]
        )
    else:
        verified_result = (
            "The result was verified successfully. Sensitive record identifiers "
            "and restricted fields have been omitted from this message."
        )
    return _customer_email(
        "We have completed the requested database review and verified the "
        "result against the connected data source.\n\n"
        + verified_result
        + "\n\nThis result reflects the data available at the time of the "
        "investigation."
    )


def _is_sensitive_customer_field(
    normalized_key: str,
    excluded_terms: set[str],
) -> bool:
    # Treat identifier fields as sensitive without hiding safe metrics such as
    # `paidAmount`, whose normalized name merely contains the letters "id".
    is_identifier = normalized_key == "id" or normalized_key.endswith("id")
    return is_identifier or any(
        term in normalized_key for term in excluded_terms
    )


def _humanize_key(value: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ")
    return spaced.strip().title()


def _format_customer_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _customer_email(body: str) -> str:
    safe_body = _redact_sensitive_text(body.strip())
    return (
        "Subject: TraceX investigation update\n\n"
        "Hello,\n\n"
        f"{safe_body}\n\n"
        "If you need any additional validation or assistance, please reply to "
        "this message and our support team will be happy to help.\n\n"
        "Regards,\n"
        "TraceX L2 Support Team"
    )


def _redact_sensitive_text(value: str) -> str:
    patterns = (
        r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|oracle)://[^\s]+",
        r"\b[a-f0-9]{24}\b",
        r"\b[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-"
        r"[89ab][a-f0-9]{3}-[a-f0-9]{12}\b",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b(?:password|secret|token|api[_ -]?key|credential)"
        r"\s*[:=]\s*[^\s,;]+",
    )
    redacted = value
    for pattern in patterns:
        redacted = re.sub(
            pattern,
            "[redacted]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def build_final_report_node(state: InvestigationState) -> dict:
    investigation = state.get("investigation")
    analysis = state.get("root_cause_analysis")
    if investigation is None:
        reason = state.get("failure_reason") or "The investigation could not be completed."
        report = RootCauseReport(
            response_type=(
                "retrieval" if _looks_like_retrieval(state) else "incident"
            ),
            verification_status="inconclusive",
            issue_summary=state["user_query"], missing_information=[reason],
            customer_response=_customer_email(
                "We reviewed the available database information but could not "
                "verify the requested result. Additional data or relationship "
                "details are required before we can provide a reliable answer. "
                "We have not made assumptions where the evidence was incomplete."
            ),
            engineering_note=(
                "The investigation did not produce a verified result. Review "
                "the missing-information section and tool execution details."
            ),
        )
    else:
        established = bool(analysis and analysis.is_established)
        retrieval_evidence = (
            _retrieval_evidence(state)
            if (
                not established
                and _looks_like_retrieval(state)
                and _retrieval_is_verified(state)
            )
            else None
        )
        is_retrieval = retrieval_evidence is not None
        missing = ([] if is_retrieval else
                   (analysis.missing_information if analysis else investigation.unresolved_questions))
        if established:
            customer = _customer_email(
                "We completed our investigation and identified the underlying "
                f"cause: {analysis.root_cause} "
                "The supporting evidence and recommended next steps are included "
                "in this report."
            )
        elif is_retrieval:
            customer = _customer_retrieval_response(
                retrieval_evidence.content.get("rows", [])
            )
        else:
            customer = _customer_email(
                "We reviewed the available evidence, but it is not sufficient "
                "to confirm a reliable conclusion at this time. The outstanding "
                "information required to continue the investigation is listed "
                "below. We recommend collecting those details before taking "
                "corrective action."
            )
        report = RootCauseReport(
            response_type=(
                "retrieval" if _looks_like_retrieval(state) else "incident"
            ),
            verification_status=(
                "verified" if is_retrieval or established else "inconclusive"
            ),
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
            engineering_note=(
                analysis.reasoning_summary
                if analysis
                else (
                    "Verified data retrieval completed; root-cause analysis "
                    "does not apply to this request."
                    if is_retrieval
                    else (
                        "The exact requested result could not be verified from "
                        "the available database evidence."
                    )
                )
            ),
            result_records=(
                []
                if _looks_like_retrieval(state) and not is_retrieval
                else _result_records(state, final_result_only=is_retrieval)
            ),
        )
    return {"final_report": report, "current_stage": "completed"}
