import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from deep_agent.models.investigation import InvestigationResult
from deep_agent.models.state import InvestigationState
from deep_agent.services.model_retry import invoke_with_rate_limit_retry
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


@lru_cache(maxsize=1)
def _model():
    load_dotenv()
    return ChatGoogleGenerativeAI(
        model=os.getenv("INVESTIGATION_MODEL", "gemini-3.1-flash-lite"), temperature=0
    ).with_structured_output(InvestigationResult)


async def investigate_node(state: InvestigationState) -> dict:
    payload = [item.model_dump(mode="json") for item in state.get("evidence", [])]
    try:
        result = await invoke_with_rate_limit_retry(
            lambda: _model().ainvoke([
                {"role": "system", "content": INVESTIGATION_PROMPT},
                {"role": "user", "content": f"Issue:\n{state['user_query']}\n\nEvidence:\n{json.dumps(payload, indent=2)}"},
            ]),
            stage="Investigation",
        )
    except Exception as exc:
        return {
            "investigation": None,
            "failure_reason": f"Investigation model error: {exc}",
            "current_stage": "investigation_failed",
        }
    valid_ids = {item.id for item in state.get("evidence", [])}
    for hypothesis in result.hypotheses:
        hypothesis.supporting_evidence_ids = [x for x in hypothesis.supporting_evidence_ids if x in valid_ids]
        hypothesis.contradicting_evidence_ids = [x for x in hypothesis.contradicting_evidence_ids if x in valid_ids]
    return {"investigation": result, "requested_evidence": result.requested_evidence,
            "current_stage": "root_cause_analysis"}


def route_after_investigation(state: InvestigationState) -> str:
    result = state.get("investigation")
    if result is None:
        return "build_inconclusive_report"
    if _is_completed_data_retrieval(state):
        return "build_result_report"
    if _looks_like_data_retrieval(state):
        if state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS:
            return "collect_more_evidence"
        return "build_inconclusive_report"
    if (result.requires_more_evidence and result.requested_evidence
            and state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS):
        return "collect_more_evidence"
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
    query = state.get("user_query", "").strip().lower()
    retrieval = bool(re.match(
        r"^(give|show|list|get|return|fetch|identify|find|which|what|who)\b",
        query,
    ))
    incident_terms = ("why", "failed", "failure", "error", "incident", "root cause")
    return retrieval and not any(term in query for term in incident_terms)
