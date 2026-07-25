import re
from typing import Any

from deep_agent.models.evidence import EvidenceType
from deep_agent.models.report import (
    ExternalReference,
    ResultRecord,
    RootCauseReport,
)
from deep_agent.models.state import InvestigationState
from deep_agent.services.retrieval_verification import final_answer_evidence


def _retrieval_evidence(state: InvestigationState):
    return final_answer_evidence(state)


def _looks_like_retrieval(state: InvestigationState) -> bool:
    understanding = state.get("query_understanding")
    if understanding is not None:
        return understanding.intent in {"data_retrieval", "informational"}
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


def _analysis_is_verified(state: InvestigationState) -> bool:
    understanding = state.get("query_understanding")
    if understanding is None or understanding.intent not in {
        "analysis", "explanation", "informational",
    }:
        return False
    investigation = state.get("investigation")
    if (
        investigation is None
        or investigation.requires_more_evidence
        or investigation.requested_evidence
        or investigation.unresolved_questions
    ):
        return False
    required_steps = [
        step
        for step in state.get("investigation_plan", [])
        if step.stage != "validation"
    ]
    return bool(required_steps) and all(
        step.status in {"completed", "blocked"}
        for step in required_steps
    ) and any(step.status == "completed" for step in required_steps)


def _analysis_evidence_ids(state: InvestigationState) -> list[str]:
    return [
        item.id
        for item in state.get("evidence", [])
        if item.evidence_type in {
            EvidenceType.CODE_REFERENCE,
            EvidenceType.DATABASE_QUERY,
            EvidenceType.DATABASE_RECORD,
            EvidenceType.DATABASE_SCHEMA,
            EvidenceType.LOG_ENTRY,
        }
        and not item.content.get("error")
    ][:20]


def _result_records(state: InvestigationState, final_result_only: bool = False) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    seen: set[str] = set()
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
            candidates = (row for row in rows if isinstance(row, dict))
        else:
            candidates = (item.content,)
        for row in candidates:
            safe_row = _safe_result_record(row)
            fingerprint = repr(sorted(safe_row.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            records.append(ResultRecord(source=item.source, record=safe_row))
        if len(records) >= 100:
            break
    return records[:100]


def _safe_result_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove credentials and direct PII from customer-visible result tables."""
    excluded_terms = {
        "password", "secret", "token", "credential", "apikey", "auth",
        "email", "phone", "mobile", "contact", "address", "billing",
    }
    return {
        key: value
        for key, value in record.items()
        if not any(
            term in re.sub(r"[^a-z0-9]", "", key.lower())
            for term in excluded_terms
        )
    }


def _customer_retrieval_response(
    rows: list[dict[str, Any]],
    state: InvestigationState | None = None,
) -> str:
    informational_note = ""
    subject = "TraceX investigation update"
    understanding = state.get("query_understanding") if state else None
    if understanding and understanding.intent == "informational":
        subject = "TraceX information request update"
        if any(term in state.get("user_query", "").lower() for term in (
            "price", "pricing", "rate", "rates", "cost", "fee", "fees",
        )):
            informational_note = (
                "\n\nPricing may vary by destination, account agreement, "
                "tax treatment, and subsequent provider updates. Please confirm "
                "the applicable destination and billing context before relying "
                "on these figures for final charges."
            )
    if len(rows) != 1:
        return _customer_email(
            "We completed the requested database review and verified "
            f"{len(rows)} matching records. The result table below contains "
            "the fields relevant to the request and reflects the data available "
            f"at the time of the investigation.{informational_note}",
            subject=subject,
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
        f"investigation.{informational_note}",
        subject=subject,
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
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return str(value)


def _customer_email(
    body: str,
    *,
    subject: str = "TraceX investigation update",
) -> str:
    safe_body = _redact_sensitive_text(body.strip())
    return (
        f"Subject: {subject}\n\n"
        "Hello,\n\n"
        f"{safe_body}\n\n"
        "If you need any additional validation or assistance, please reply to "
        "this message and our support team will be happy to help.\n\n"
        "Regards,\n"
        "TraceX L2 Support Team"
    )


def _customer_safe_text(value: str, state: InvestigationState) -> str:
    """Remove implementation details that belong only in engineering notes."""
    result = value
    internal_names = {
        item.source.strip()
        for item in state.get("evidence", [])
        if item.source and item.source.strip()
    }
    for source in sorted(internal_names, key=len, reverse=True):
        if len(source) >= 3:
            result = re.sub(
                re.escape(source),
                "the connected data source",
                result,
                flags=re.IGNORECASE,
            )
    result = re.sub(
        r"\b(?:collection|table|schema)\s+['\"`][^'\"`]+['\"`]",
        "connected data source",
        result,
        flags=re.IGNORECASE,
    )
    return _redact_sensitive_text(result)


def _external_references(
    state: InvestigationState,
) -> list[ExternalReference]:
    references: list[ExternalReference] = []
    seen: set[str] = set()
    for item in state.get("evidence", []):
        content = item.content
        if (
            content.get("external_context_only") is not True
            or content.get("unavailable")
            or content.get("error")
        ):
            continue
        candidates = content.get("citations", [])
        if content.get("url"):
            candidates = [
                {
                    "title": content.get("title") or "Official documentation",
                    "url": content["url"],
                },
                *candidates,
            ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            url = str(candidate.get("url", "")).strip()
            if not url.startswith("https://") or url in seen:
                continue
            seen.add(url)
            references.append(ExternalReference(
                title=(
                    str(candidate.get("title", "")).strip()
                    or "Official documentation"
                )[:300],
                url=url,
            ))
            if len(references) >= 5:
                return references
    return references


def _append_customer_references(
    message: str,
    references: list[ExternalReference],
) -> str:
    if not references:
        return message
    block = "Reference documentation:\n" + "\n".join(
        f"- {item.title}: {item.url}" for item in references
    )
    marker = (
        "\n\nIf you need any additional validation or assistance, "
        "please reply to this message"
    )
    if marker not in message:
        return f"{message}\n\n{block}"
    return message.replace(
        marker,
        f"\n\n{block}{marker}",
        1,
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
    references = _external_references(state)
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
        understanding = state.get("query_understanding")
        is_incident_request = bool(
            understanding is None
            or understanding.intent == "incident_investigation"
        )
        established = bool(
            is_incident_request and analysis and analysis.is_established
        )
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
        is_analysis = (
            not established and not is_retrieval
            and _analysis_is_verified(state)
        )
        missing = ([] if is_retrieval or is_analysis else
                   (analysis.missing_information if analysis else investigation.unresolved_questions))
        if established:
            resolution_steps = (
                analysis.recommended_fix
                or analysis.suggested_actions
            )
            resolution = ""
            if resolution_steps:
                resolution = (
                    "\n\nRecommended resolution:\n"
                    + "\n".join(
                        f"- {step}" for step in resolution_steps[:5]
                    )
                )
            customer = _customer_email(
                "We completed our investigation and identified the underlying "
                f"cause:\n\n{analysis.root_cause}"
                f"{resolution}\n\n"
                "No customer data was modified during this investigation."
            )
        elif is_retrieval:
            customer = _customer_retrieval_response(
                retrieval_evidence.content.get("rows", []),
                state,
            )
        elif is_analysis:
            customer = _customer_email(
                "We completed the requested review and verified the available "
                "information against the connected sources.\n\n"
                "Verified findings:\n"
                f"{_customer_safe_text(investigation.actual_state, state)}\n\n"
                "These findings reflect the information available at the time "
                "of the review. Where pricing, limits, or availability may vary "
                "by region or account, the applicable scope should be confirmed "
                "before relying on the result."
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
                "informational"
                if understanding and understanding.intent == "informational"
                else "retrieval"
                if _looks_like_retrieval(state)
                else "explanation"
                if is_analysis
                else "incident"
            ),
            verification_status=(
                "verified"
                if is_retrieval or is_analysis or established
                else "inconclusive"
            ),
            issue_summary=(
                investigation.issue_summary
                if established or is_retrieval or is_analysis
                else (
                    "The reported incident could not be conclusively diagnosed "
                    "from the available evidence."
                )
            ),
            root_cause=analysis.root_cause if established else None,
            confidence=analysis.confidence if analysis else 0.0,
            supporting_evidence_ids=(analysis.supporting_evidence_ids if analysis else
                                     ([retrieval_evidence.id] if retrieval_evidence else
                                      _analysis_evidence_ids(state) if is_analysis else [])),
            contradicting_evidence_ids=analysis.contradicting_evidence_ids if analysis else [],
            affected_components=investigation.affected_components,
            expected_state=investigation.expected_state,
            actual_state=(
                investigation.actual_state
                if established or is_retrieval or is_analysis
                else (
                    "The reported failure was not reproduced or causally "
                    "verified. Runtime error evidence is required before "
                    "identifying a corrective action."
                )
            ),
            suggested_actions=analysis.suggested_actions if analysis else [],
            missing_information=missing,
            customer_response=customer,
            engineering_note=(
                (
                    analysis.reasoning_summary
                    + (
                        "\n\nContributing factors: "
                        + "; ".join(analysis.contributing_factors)
                        if analysis.contributing_factors
                        else ""
                    )
                )
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
                _result_records(state, final_result_only=True)
                if is_retrieval
                else []
            ),
            investigation_status=(
                "resolved"
                if established or is_retrieval or is_analysis
                else "insufficient_evidence"
            ),
            impact=(
                ", ".join(investigation.affected_components)
                if investigation.affected_components
                else "Impact could not be quantified from available evidence."
            ),
            investigation_steps=[
                item.objective
                for item in state.get("investigation_plan", [])
            ],
            rejected_hypotheses=[
                item.statement
                for item in investigation.hypotheses
                if item.status in {"rejected", "contradicted"}
            ],
            contributing_factors=(
                analysis.contributing_factors if analysis else []
            ),
            recommended_fix=(
                analysis.recommended_fix or analysis.suggested_actions
                if established and analysis
                else []
            ),
            validation_steps=(
                analysis.validation_steps
                if established and analysis and analysis.validation_steps
                else [
                    step
                    for item in investigation.hypotheses
                    for step in item.validation_steps
                ][:10]
            ),
            risks=(
                ["Conclusion remains unverified until missing evidence is collected."]
                if not established and not is_retrieval and not is_analysis
                else []
            ),
            external_references=references,
        )
    report.external_references = references
    report.customer_response = _append_customer_references(
        report.customer_response,
        references,
    )
    return {"final_report": report, "current_stage": "completed"}
