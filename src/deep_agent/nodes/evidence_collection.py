from deep_agent.agents.evidence_agent import create_evidence_agent
from langchain_core.runnables import RunnableConfig
from deep_agent.models.state import InvestigationState
from deep_agent.services.model_retry import is_rate_limit_error, invoke_with_rate_limit_retry
from deep_agent.services.evidence_repository import (
    bind_investigation,
    evidence_repository,
    reset_investigation,
)


async def collect_evidence_node(
    state: InvestigationState, config: RunnableConfig | None = None
) -> dict:
    investigation_id = state["investigation_id"]
    prompt = (
        f"Issue:\n{state['user_query']}\n\n"
        f"Organization: {state['organization_id']}\nProject: {state['project_id']}\n"
        f"Additional evidence requested: {state.get('requested_evidence') or 'None'}\n"
        "Inspect schemas before records. Collect facts only; do not determine root cause."
    )
    token = bind_investigation(investigation_id)
    try:
        agent = create_evidence_agent()
        await invoke_with_rate_limit_retry(
            lambda: agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {
                    "investigation_id": investigation_id,
                    "organization_id": state["organization_id"],
                    "project_id": state["project_id"],
                }, "callbacks": (config or {}).get("callbacks")},
            ),
            stage="Evidence collection",
        )
        evidence = await evidence_repository.list_by_investigation(investigation_id)
        return {
            "evidence": evidence,
            "evidence_collection_attempts": state.get("evidence_collection_attempts", 0) + 1,
            "evidence_collection_errors": [],
            "requested_evidence": [],
            "current_stage": "evidence_validation",
        }
    except Exception as exc:
        errors = [*state.get("evidence_collection_errors", []), str(exc)]
        attempts = state.get("evidence_collection_attempts", 0) + 1
        # The node already honored RetryInfo internally. Do not let the graph's
        # evidence loop multiply a persistent quota failure into nine attempts.
        if is_rate_limit_error(exc):
            attempts = 3
        return {
            "evidence_collection_attempts": attempts,
            "evidence_collection_errors": errors,
            "current_stage": "evidence_collection_failed",
        }
    finally:
        reset_investigation(token)
