import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.models.state import InvestigationState
from deep_agent.services.model_retry import invoke_with_rate_limit_retry

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
"""


@lru_cache(maxsize=1)
def _model():
    load_dotenv()
    return ChatGoogleGenerativeAI(
        model=os.getenv("ROOT_CAUSE_MODEL", "gemini-3.1-flash-lite"), temperature=0
    ).with_structured_output(RootCauseAnalysis)


async def identify_root_cause_node(state: InvestigationState) -> dict:
    evidence = [item.model_dump(mode="json") for item in state.get("evidence", [])]
    try:
        result = await invoke_with_rate_limit_retry(
            lambda: _model().ainvoke([
                {"role": "system", "content": ROOT_CAUSE_PROMPT},
                {"role": "user", "content": json.dumps({
                    "issue": state["user_query"],
                    "investigation": state["investigation"].model_dump(mode="json"),
                    "evidence": evidence,
                }, indent=2)},
            ]),
            stage="Root-cause analysis",
        )
    except Exception as exc:
        return {
            "root_cause_analysis": None,
            "failure_reason": f"Root-cause model error: {exc}",
            "current_stage": "root_cause_failed",
        }
    return {"root_cause_analysis": result, "current_stage": "root_cause_validation"}
