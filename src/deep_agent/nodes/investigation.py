import json
import re

from deep_agent.models.investigation import InvestigationResult
from deep_agent.models.state import InvestigationState
from deep_agent.services.evidence_context import compact_evidence
from deep_agent.services.reasoning import reasoning_call_limit, reasoning_service
from deep_agent.stage_prompts import (
    INVESTIGATION_PROMPT as STAGE_INVESTIGATION_PROMPT,
)
from deep_agent.services.retrieval_verification import final_answer_evidence

MAX_COLLECTION_ATTEMPTS = 3
INVESTIGATION_PROMPT = """
You are TraceX's Principal Investigation Agent.

You receive ONLY tool-persisted evidence.

You do not collect evidence.
You do not determine the final root cause.

Your responsibility is to determine whether the collected evidence supports a
complete and defensible investigation.

====================================================================
OBJECTIVES
====================================================================

1. Validate evidence quality.
2. Verify completeness.
3. Compare expected vs actual state.
4. Build competing hypotheses.
5. Identify contradictions.
6. Request missing evidence.
7. Produce an investigation report.

====================================================================
RULES
====================================================================

1. Never invent facts.
2. Never infer unsupported relationships.
3. Never assume business logic.
4. Never use external knowledge.
5. Every conclusion must reference evidence IDs.

====================================================================
DATA RETRIEVAL VALIDATION
====================================================================

Validate:

- Filters
- Sorting
- Aggregations
- Limits
- Joins
- Relationship keys

Ensure:

- Requested entity was returned.
- A human-readable entity value is included when requested; an ID alone is not
  sufficient for a name-based request.
- Decisive metric is present.
- Supporting identifiers exist.
- Final query succeeded.
- Final query evidence is explicitly marked evidence_role="final_answer".
- Expected cardinality is satisfied and unrelated fallback rows are excluded.

Examples:

Incorrect:
User:
"Highest revenue customer"

Returned:
payment_id

Correct:
customer_id
customer_name
revenue
currency

====================================================================
INCIDENT INVESTIGATION
====================================================================

Determine:

Expected State:
- What should happen?

Actual State:
- What happened?

Timeline:
- When did divergence begin?

Scope:
- Who is affected?

Impact:
- Severity and blast radius.

====================================================================
HYPOTHESIS GENERATION
====================================================================

Generate up to three competing hypotheses.

Each hypothesis must include:

- Supporting evidence IDs
- Contradicting evidence IDs
- Confidence score
- Missing evidence

Confidence:

- LOW
- MEDIUM
- HIGH

Correlation does not imply causation.

An exact customer-reported system error may establish the rejected business
condition when it is corroborated by matching database, code, API, or log
evidence. Do not discard an explicit entitlement or validation error merely
because runtime logs are unavailable. Clearly distinguish the reported message
from independently verified supporting state.

====================================================================
MISSING EVIDENCE
====================================================================

If evidence is insufficient:

Set:

requires_more_evidence = true

Provide exactly one targeted request.

Examples:

- Verify booking->payment relationship.
- Retrieve failed webhook payload.
- Fetch transaction status history.
- Confirm enum mapping.
- Retrieve audit logs.

====================================================================
SELF-CORRECTION
====================================================================

Detect:

- Missing joins
- Missing filters
- Incorrect ordering
- Wrong entity returned
- Incomplete timelines

Do not proceed if critical evidence is missing.

====================================================================
OUTPUT
====================================================================

Produce:

- issue_summary
- expected_state
- actual_state
- timeline
- impact
- hypotheses
- contradictions
- unresolved_questions
- confidence
- requires_more_evidence

Prefer:

"Inconclusive due to insufficient evidence."

over:

An unsupported conclusion.
"""


async def investigate_node(state: InvestigationState) -> dict:
    reasoning_calls = state.get("reasoning_calls", 0)
    if reasoning_calls >= reasoning_call_limit():
        return {
            "investigation": None,
            "failure_reason": "Investigation reasoning budget was exhausted.",
            "current_stage": "investigation_failed",
        }
    payload = compact_evidence(state.get("evidence", []))
    try:
        result = await reasoning_service().invoke(
            InvestigationResult,
            [
                {"role": "system", "content": STAGE_INVESTIGATION_PROMPT},
                {"role": "user", "content": f"Issue:\n{state['user_query']}\n\nEvidence:\n{json.dumps(payload, indent=2)}"},
            ],
            stage="Investigation",
        )
    except Exception as exc:
        return {
            "investigation": None,
            "failure_reason": f"Investigation model error: {exc}",
            "current_stage": "investigation_failed",
            "reasoning_calls": reasoning_calls + 1,
        }
    valid_ids = {item.id for item in state.get("evidence", [])}
    for hypothesis in result.hypotheses:
        hypothesis.supporting_evidence_ids = [x for x in hypothesis.supporting_evidence_ids if x in valid_ids]
        hypothesis.contradicting_evidence_ids = [x for x in hypothesis.contradicting_evidence_ids if x in valid_ids]
    # Do not allow executable validation work to leak into the final report as
    # an unperformed suggestion. Turn unresolved questions and hypothesis
    # validation steps into a targeted evidence-collection request while the
    # workflow still has collection attempts available.
    if not result.requested_evidence:
        follow_ups = [
            *result.unresolved_questions,
            *[
                step
                for hypothesis in result.hypotheses
                if hypothesis.status != "contradicted"
                for step in hypothesis.validation_steps
            ],
        ]
        result.requested_evidence = list(dict.fromkeys(
            item.strip() for item in follow_ups if item.strip()
        ))[:3]
    understanding = state.get("query_understanding")
    if (
        understanding
        and understanding.intent in {"data_retrieval", "informational"}
        and final_answer_evidence(state) is None
    ):
        result.requested_evidence = [
            "Run one conclusive database query with purpose='final_answer' "
            "that applies the verified entity filter, requested ordering, "
            "cardinality, and output fields. Do not reuse an exploration "
            "sample as the answer."
        ]
    if result.requested_evidence:
        result.requires_more_evidence = True
    return {"investigation": result, "requested_evidence": result.requested_evidence,
            "reasoning_calls": reasoning_calls + 1,
            "current_stage": "root_cause_analysis"}


def route_after_investigation(state: InvestigationState) -> str:
    result = state.get("investigation")
    if result is None:
        return "build_inconclusive_report"
    understanding = state.get("query_understanding")
    is_incident = bool(
        understanding
        and understanding.intent == "incident_investigation"
    )
    if _is_completed_data_retrieval(state):
        return "build_result_report"
    if _looks_like_data_retrieval(state):
        if state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS:
            return "collect_more_evidence"
        return "build_inconclusive_report"
    if (result.requires_more_evidence and result.requested_evidence
            and state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS):
        return "collect_more_evidence"
    if not is_incident:
        return (
            "build_inconclusive_report"
            if result.requires_more_evidence or result.unresolved_questions
            else "build_result_report"
        )
    return "identify_root_cause"


def _is_completed_data_retrieval(state: InvestigationState) -> bool:
    retrieval = _looks_like_data_retrieval(state)
    if not retrieval:
        return False
    result = state.get("investigation")
    if result is None or result.requires_more_evidence or result.requested_evidence:
        return False
    unresolved_text = " ".join([
        result.actual_state,
        *result.unresolved_questions,
    ]).lower()
    if any(marker in unresolved_text for marker in (
        "not explicitly defined",
        "not established",
        "cannot determine",
        "could not determine",
        "unable to determine",
        "missing relationship",
        "relationship is unknown",
        "insufficient evidence",
    )):
        return False
    return final_answer_evidence(state) is not None


def _looks_like_data_retrieval(state: InvestigationState) -> bool:
    understanding = state.get("query_understanding")
    if understanding is not None:
        return understanding.intent in {"data_retrieval", "informational"}
    query = state.get("user_query", "").strip().lower()
    retrieval = bool(re.match(
        r"^(give|show|list|get|return|fetch|identify|find|which|what|who)\b",
        query,
    ))
    incident_terms = ("why", "failed", "failure", "error", "incident", "root cause")
    return retrieval and not any(term in query for term in incident_terms)
