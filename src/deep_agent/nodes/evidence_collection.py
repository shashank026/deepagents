import asyncio
import os
from datetime import datetime, timezone

from deep_agent.agents.evidence_agent import create_evidence_agent
from deep_agent.models.evidence import (
    Evidence,
    EvidenceReliability,
    EvidenceType,
)
from langchain_core.runnables import RunnableConfig
from deep_agent.models.state import InvestigationState
from deep_agent.models.query import EvidenceSource
from deep_agent.tools.web_research import web_research_enabled
from deep_agent.services.model_retry import is_rate_limit_error, invoke_with_rate_limit_retry
from deep_agent.services.evidence_repository import (
    bind_investigation,
    evidence_repository,
    reset_investigation,
)
from deep_agent.services.evidence_context import compact_evidence
from deep_agent.services.skills import select_skills


async def collect_evidence_node(
    state: InvestigationState, config: RunnableConfig | None = None
) -> dict:
    investigation_id = state["investigation_id"]
    understanding = state.get("query_understanding")
    source_plan = state.get("evidence_source_plan")
    database_inventory = [
        {
            "connection_id": source.get("connection_id"),
            "database_type": source.get("provider")
                or source.get("analysis", {}).get("database_type"),
        }
        for source in state.get("database_sources", [])
    ]
    prior_evidence = await evidence_repository.list_by_investigation(
        investigation_id
    )
    # A durable LangGraph checkpoint may outlive this process-local cache.
    # Rehydrate the cache from checkpointed state before continuing a resumed run.
    for item in state.get("evidence", []):
        await evidence_repository.save(investigation_id, item)
    prior_evidence = await evidence_repository.list_by_investigation(investigation_id)
    if not any(
        item.evidence_type == EvidenceType.USER_INPUT
        for item in prior_evidence
    ):
        reported = Evidence(
            id=f"ev-user-input-{investigation_id}",
            evidence_type=EvidenceType.USER_INPUT,
            source="customer_report",
            summary="Customer-reported incident details",
            content={
                "reported_text": state["user_query"],
                "provenance": "customer_supplied",
                "independently_verified": False,
            },
            reliability=EvidenceReliability.MEDIUM,
            collected_at=datetime.now(timezone.utc).isoformat(),
        )
        await evidence_repository.save(investigation_id, reported)
        prior_evidence = [*prior_evidence, reported]
    prior_summary = compact_evidence(prior_evidence, max_items=18)
    planned_sources = set(source_plan.sources) if source_plan else {EvidenceSource.DATABASE}
    active_skills = select_skills(understanding, planned_sources)
    skill_context = [
        {"name": skill.name, "instructions": skill.instructions}
        for skill in active_skills
    ]
    investigation_plan = [
        item.model_dump(mode="json")
        for item in state.get("investigation_plan", [])
    ]
    organization_knowledge = state.get("organization_knowledge", [])[:5]
    prompt = (
        f"Issue:\n{state['user_query']}\n\n"
        f"Additional evidence requested: {state.get('requested_evidence') or 'None'}\n"
        f"Extracted business entities: "
        f"{understanding.model_dump(mode='json') if understanding else {}}\n"
        f"Required evidence sources: "
        f"{source_plan.model_dump(mode='json') if source_plan else {'sources': ['database']}}\n"
        f"Investigation plan: {investigation_plan}\n"
        f"Connected database types (authoritative): {database_inventory}\n"
        f"Execution context manifest (authoritative capabilities and limits): "
        f"{state.get('context_manifest', {})}\n"
        f"Relevant organization-scoped prior learnings (planning hints only; "
        f"not evidence and must be independently revalidated): "
        f"{organization_knowledge}\n"
        f"Evidence already collected (continue from it; do not repeat): {prior_summary}\n"
        f"Applicable investigation skills (loaded on demand): {skill_context}\n"
        "Inspect schemas before records. Collect facts only; do not determine root cause. "
        "TraceX organization and project identifiers are control-plane metadata and "
        "must never be used as values in client database filters or queries."
    )
    token = bind_investigation(investigation_id)
    try:
        sources = set(source_plan.sources) if source_plan else None
        # Add public research on a later round when internal analysis requests
        # it, even if it was not part of the initial plan. Planned public
        # research (for example pricing or provider documentation) remains
        # available in the first round.
        if (
            sources is not None
            and web_research_enabled()
            and state.get("evidence_collection_attempts", 0) > 0
            and state.get("requested_evidence")
            and any(
                item.evidence_type != EvidenceType.USER_INPUT
                for item in prior_evidence
            )
        ):
            sources.add(EvidenceSource.WEB)
        selected_groups = (
            [{source} for source in sorted(sources, key=lambda item: item.value)]
            if sources and len(sources) > 1
            else [sources]
        )
        semaphore = asyncio.Semaphore(max(
            1, int(os.getenv("EVIDENCE_SOURCE_MAX_CONCURRENCY", "3"))
        ))

        async def collect_group(group):
            async with semaphore:
                agent = create_evidence_agent(group)
                label = ",".join(item.value for item in group) if group else "default"
                return await invoke_with_rate_limit_retry(
                    lambda: agent.ainvoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config={"configurable": {
                            "investigation_id": investigation_id,
                            "organization_id": state["organization_id"],
                            "project_id": state["project_id"],
                        }, "callbacks": (config or {}).get("callbacks"),
                            "recursion_limit": 30},
                    ),
                    stage=f"Evidence collection ({label})",
                )

        results = await asyncio.gather(
            *(collect_group(group) for group in selected_groups),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures and len(failures) == len(results):
            raise failures[0]
        evidence = await evidence_repository.list_by_investigation(investigation_id)
        return {
            "evidence": evidence,
            "evidence_collection_attempts": state.get("evidence_collection_attempts", 0) + 1,
            "evidence_collection_errors": [str(item) for item in failures],
            "requested_evidence": [],
            "current_stage": "evidence_validation",
        }
    except Exception as exc:
        evidence = await evidence_repository.list_by_investigation(investigation_id)
        errors = [*state.get("evidence_collection_errors", []), str(exc)]
        attempts = state.get("evidence_collection_attempts", 0) + 1
        # The node already honored RetryInfo internally. Do not let the graph's
        # evidence loop multiply a persistent quota failure into nine attempts.
        if is_rate_limit_error(exc):
            attempts = 3
        return {
            "evidence": evidence,
            "evidence_collection_attempts": attempts,
            "evidence_collection_errors": errors,
            "current_stage": "evidence_collection_failed",
        }
    finally:
        reset_investigation(token)
