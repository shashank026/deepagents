from deep_agent.models.evidence import EvidenceType
from deep_agent.models.execution import FailedAssumption, ToolError
from deep_agent.models.state import InvestigationState
from deep_agent.nodes.investigation import route_after_investigation

MAX_QUERY_RETRIES = 3
MAX_TOOL_RETRIES = 3
MAX_HYPOTHESIS_REVISIONS = 3


def self_check_node(state: InvestigationState) -> dict:
    failed = list(state.get("failed_assumptions", []))
    errors = list(state.get("tool_errors", []))
    requested = list(state.get("requested_evidence", []))
    evidence = state.get("evidence", [])

    for step in state.get("investigation_plan", []):
        if step.stage != "validation" and step.status == "pending":
            requested.append(
                f"Complete the {step.stage} research requirement: "
                f"{step.objective} Completion criteria: "
                f"{step.completion_criteria}"
            )

    for item in evidence:
        error = item.content.get("error")
        if error:
            if item.evidence_type == EvidenceType.DATABASE_QUERY:
                correction = (
                    "Re-inspect schema and retry with "
                    "execute_typed_database_query using the exact "
                    "analyzed collection and field names and scalar filter "
                    "values; the schema compiler handles native database types."
                )
            else:
                correction = (
                    "Re-inspect the connected source and retry with changed inputs."
                )
            errors.append(ToolError(
                tool_name=_tool_name(item),
                error_code=type(error).__name__.upper(),
                error_message=str(error),
                retryable=_retryable_error(str(error)),
                input_summary=_input_summary(item.content),
            ))
            failed.append(FailedAssumption(
                assumption=item.summary,
                reason=str(error),
                evidence_ids=[item.id],
                correction=correction,
                retryable=_retryable_error(str(error)),
            ))
        if (
            item.evidence_type == EvidenceType.DATABASE_QUERY
            and item.content.get("evidence_role") == "final_answer"
            and item.content.get("row_count") == 0
        ):
            failed.append(FailedAssumption(
                assumption="The selected final filter/value would return the requested data.",
                reason="The final query returned zero rows.",
                evidence_ids=[item.id],
                correction=(
                    "Inspect schema-native field types and representative stored "
                    "values, then revise the filter without changing tenant scope."
                ),
            ))
            requested.append(
                "Inspect schema-native types and representative stored values "
                "before retrying the empty final query."
            )

    failed = _dedupe(failed, lambda item: (item.assumption, item.reason))
    errors = _dedupe(
        errors,
        lambda item: (
            item.tool_name, item.error_code, item.error_message,
            repr(sorted(item.input_summary.items())),
        ),
    )
    return {
        "failed_assumptions": failed,
        "tool_errors": errors,
        "requested_evidence": list(dict.fromkeys(requested))[:3],
        "current_stage": "self_check",
    }


def route_after_self_check(state: InvestigationState) -> str:
    route = route_after_investigation(state)
    return "revise_investigation" if route == "collect_more_evidence" else route


def revise_investigation_node(state: InvestigationState) -> dict:
    retries = dict(state.get("retry_counts", {}))
    retries["hypothesis"] = retries.get("hypothesis", 0) + 1
    requested = list(state.get("requested_evidence", []))
    for item in state.get("failed_assumptions", []):
        if item.retryable and item.correction:
            requested.append(item.correction)
    source_plan = state.get("evidence_source_plan")
    if source_plan:
        requested_text = " ".join(requested).lower()
        additions = []
        for source in source_plan.optional_sources:
            source_terms = {
                "codebase": ("code", "implementation", "mapping", "decision path"),
                "logs": ("log", "runtime", "timeline", "event"),
                "web": ("documentation", "specification", "release note"),
            }.get(source.value, (source.value,))
            if any(term in requested_text for term in source_terms):
                additions.append(source)
        source_plan = source_plan.model_copy(update={
            "sources": list(dict.fromkeys([*source_plan.sources, *additions])),
            "optional_sources": [
                item for item in source_plan.optional_sources if item not in additions
            ],
        })
    return {
        "retry_counts": retries,
        "requested_evidence": list(dict.fromkeys(requested))[:3],
        "evidence_source_plan": source_plan,
        "current_stage": "revise_investigation",
    }


def _tool_name(item) -> str:
    if item.evidence_type == EvidenceType.DATABASE_QUERY:
        return "database_query"
    if item.evidence_type == EvidenceType.CODE_REFERENCE:
        return "codebase"
    if item.evidence_type == EvidenceType.LOG_ENTRY:
        return "logs"
    return item.source


def _input_summary(content: dict) -> dict:
    return {
        key: content.get(key)
        for key in (
            "query", "collection", "filter", "projection", "sort", "pipeline",
            "path", "ref", "sha",
        )
        if key in content
    }


def _retryable_error(message: str) -> bool:
    lowered = message.lower()
    return not any(term in lowered for term in (
        "permission", "authentication", "authorization", "unavailable",
        "not configured",
    ))


def _dedupe(items, key):
    result = {}
    for item in items:
        result[key(item)] = item
    return list(result.values())
