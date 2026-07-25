import json

from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.models.state import InvestigationState
from deep_agent.services.evidence_context import compact_evidence
from deep_agent.services.reasoning import reasoning_call_limit, reasoning_service
from deep_agent.stage_prompts import (
    ROOT_CAUSE_PROMPT as STAGE_ROOT_CAUSE_PROMPT,
)

ROOT_CAUSE_PROMPT = """
You are TraceX's Principal Root Cause Analysis Agent.

You are the final analytical stage.

Your responsibility is to determine whether a causal mechanism has been
established from investigation evidence.

====================================================================
RULES
====================================================================

1. Use ONLY investigation output.
2. Use ONLY evidence IDs.
3. Never introduce new evidence.
4. Never speculate.
5. Never infer missing relationships.
6. Root cause requires causation.

====================================================================
ROOT CAUSE REQUIREMENTS
====================================================================

A root cause exists only if all are true:

1. Event sequence is established.
2. Supporting evidence exists.
3. Contradictions are addressed.
4. Causal mechanism is identified.
5. Alternative hypotheses are rejected.

For deterministic validation or entitlement failures, an exact
customer-reported system error can establish the rejected condition when at
least one independent source confirms the matching account, project,
subscription, plan, feature gate, or code path. Runtime reproduction is not
mandatory in that case. Never treat customer text alone as sufficient proof.

If any condition is false:

is_established = false
root_cause = null

====================================================================
VALID ROOT CAUSE EXAMPLES
====================================================================

GOOD:

"Payment failures occurred because the payment gateway returned
AUTHENTICATION_FAILED after credential rotation at 14:03 UTC."

BAD:

"Payment gateway looks suspicious."

GOOD:

"Booking creation failed because webhook processing stopped after Redis
connectivity loss."

BAD:

"Redis may be involved."

====================================================================
DATA RETRIEVAL REQUESTS
====================================================================

If the original request is:

- Highest customer
- Latest payment
- Largest booking

Then:

is_established = false
root_cause = null

Root cause analysis only applies to incidents.

====================================================================
CONTRADICTIONS
====================================================================

You must explicitly account for:

- Contradicting evidence.
- Alternative explanations.
- Missing evidence.

A contradiction that cannot be resolved invalidates the root cause.

====================================================================
CONFIDENCE
====================================================================

Assign:

- LOW
- MEDIUM
- HIGH

HIGH confidence requires:

- Complete evidence.
- No contradictions.
- Verified causality.

====================================================================
OUTPUT
====================================================================

Return:

- is_established
- root_cause
- causal_mechanism
- supporting_evidence_ids
- contradicting_evidence_ids
- rejected_hypotheses
- confidence
- reasoning_summary

Your reasoning must be concise, evidence-based, and suitable for an internal L2 support report.

When evidence is insufficient, prefer:

"Root cause could not be established."

over any speculative explanation.

Suggested actions are allowed only for work that could not be performed with
the connected tools because a source, permission, or required datum was
unavailable. Never list an action as suggested if the evidence shows it was
already completed, and never imply that a suggested action was performed.
"""


async def identify_root_cause_node(state: InvestigationState) -> dict:
    reasoning_calls = state.get("reasoning_calls", 0)
    if reasoning_calls >= reasoning_call_limit():
        return {
            "root_cause_analysis": None,
            "failure_reason": "Root-cause reasoning budget was exhausted.",
            "current_stage": "root_cause_failed",
        }
    investigation = state["investigation"]
    referenced_ids = {
        evidence_id
        for hypothesis in investigation.hypotheses
        for evidence_id in (
            hypothesis.supporting_evidence_ids
            + hypothesis.contradicting_evidence_ids
        )
    }
    evidence = compact_evidence(
        state.get("evidence", []),
        max_items=18,
        referenced_ids=referenced_ids,
    )
    try:
        result = await reasoning_service().invoke(
            RootCauseAnalysis,
            [
                {"role": "system", "content": STAGE_ROOT_CAUSE_PROMPT},
                {"role": "user", "content": json.dumps({
                    "issue": state["user_query"],
                    "investigation": investigation.model_dump(mode="json"),
                    "evidence": evidence,
                }, indent=2)},
            ],
            stage="Root-cause analysis",
        )
    except Exception as exc:
        return {
            "root_cause_analysis": None,
            "failure_reason": f"Root-cause model error: {exc}",
            "current_stage": "root_cause_failed",
            "reasoning_calls": reasoning_calls + 1,
        }
    return {
        "root_cause_analysis": result,
        "current_stage": "root_cause_validation",
        "reasoning_calls": reasoning_calls + 1,
    }
