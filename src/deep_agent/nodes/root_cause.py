import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.models.state import InvestigationState
from deep_agent.services.model_retry import invoke_with_rate_limit_retry

ROOT_CAUSE_PROMPT = """Select only a hypothesis from the investigation and use
only supplied evidence IDs. Account for contradictions. A root cause requires a
causal mechanism, not correlation. Set is_established=false and root_cause=null
when evidence is insufficient or when the request is data retrieval rather than
an incident. Return only a concise evidence-based reasoning summary."""


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
