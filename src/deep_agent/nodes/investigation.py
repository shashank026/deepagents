import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from deep_agent.models.investigation import InvestigationResult
from deep_agent.models.state import InvestigationState
from deep_agent.services.model_retry import invoke_with_rate_limit_retry

MAX_COLLECTION_ATTEMPTS = 3
INVESTIGATION_PROMPT = """Analyze only supplied evidence. Compare expected and
observed state, create competing hypotheses, link all claims to valid evidence
IDs, note contradictions, and request more evidence when needed. Do not decide
the root cause or treat correlation as causation. Prefer inconclusive results to
unsupported conclusions."""


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
    if (result.requires_more_evidence and result.requested_evidence
            and state.get("evidence_collection_attempts", 0) < MAX_COLLECTION_ATTEMPTS):
        return "collect_more_evidence"
    return "identify_root_cause"


def _is_completed_data_retrieval(state: InvestigationState) -> bool:
    query = state.get("user_query", "").strip().lower()
    retrieval = bool(re.match(r"^(give|show|list|get|return|fetch)\b", query))
    incident_terms = ("why", "failed", "failure", "error", "incident", "root cause")
    if not retrieval or any(term in query for term in incident_terms):
        return False
    return any(
        isinstance(item.content.get("rows"), list) and item.content["rows"]
        for item in state.get("evidence", [])
    )
