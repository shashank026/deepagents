import asyncio
import pytest

from deep_agent.models.evidence import Evidence, EvidenceType
from deep_agent.models.investigation import Hypothesis, InvestigationResult
from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.nodes.evidence_validation import route_after_evidence_validation, validate_evidence_node
from deep_agent.nodes import evidence_collection
from deep_agent.nodes.investigation import route_after_investigation
from deep_agent.nodes.report_builder import build_final_report_node
from deep_agent.nodes.root_cause_validation import validate_root_cause_node
from deep_agent.nodes.query_understanding import extract_business_entities_node
from deep_agent.nodes.context_manifest import build_context_manifest_node
from deep_agent.nodes.source_planning import plan_evidence_sources_node
from deep_agent.models.query import EvidenceSource
from deep_agent.models.query import TypedQueryIntent
from deep_agent.models.execution import InvestigationPlanStep
from deep_agent.agents import evidence_agent
from deep_agent import api as deepagents_api
from deep_agent import main as deepagents_main
from deep_agent.services.model_retry import is_rate_limit_error, retry_delay_seconds
from deep_agent.services.database_context import bind_database_sources, reset_database_sources
from deep_agent.services.evidence_repository import (
    bind_investigation,
    evidence_repository,
    reset_investigation,
)
from deep_agent.services.retrieval_verification import final_answer_evidence
from deep_agent.services.evidence_context import compact_evidence
from deep_agent.services.structured_output import invoke_structured
from deep_agent.services.reasoning import ReasoningService
from deep_agent.nodes.self_check import self_check_node
from deep_agent.nodes.report_validation import validate_report_node
from deep_agent.nodes.report_validation import _redact as redact_report_text
from deep_agent.tools import evidence_tools
from deep_agent.tools.database import (
    apply_limit,
    validate_read_only_query,
    _reject_control_plane_values,
    _normalize_sort_direction,
    _normalize_mongodb_sort,
    _bounded_mongodb_pipeline,
    _coerce_mongodb_ids,
    _coerce_mongodb_schema_types,
    _normalize_mongodb_extended_json,
    _validate_mongodb_value,
)
from deep_agent.tools import database as database_tools
from deep_agent.tools.tools import retrieve_schema_context
from deep_agent.services.skills import select_skills
from deep_agent.services.checkpointing import checkpoint_provider
from deep_agent.models.query import EvidenceSourcePlan
from deep_agent.tools.external_sources import SourceUnavailableError


def evidence():
    return Evidence(id="ev-1", evidence_type=EvidenceType.DATABASE_QUERY,
                    source="db", summary="one row",
                    content={
                        "rows": [{"id": 1}],
                        "row_count": 1,
                        "evidence_role": "final_answer",
                        "error": None,
                    })


def investigation():
    return InvestigationResult(issue_summary="issue", expected_state="ok", actual_state="bad",
        hypotheses=[Hypothesis(id="h-1", statement="cause", confidence=.9,
                               status="supported", supporting_evidence_ids=["ev-1"])])


def test_evidence_validation_deduplicates_and_routes():
    update = validate_evidence_node({"evidence": [evidence(), evidence()]})
    assert len(update["evidence"]) == 1
    assert route_after_evidence_validation({"evidence": update["evidence"]}) == "investigate"


def test_final_response_accepts_langgraph_dict_report():
    payload = deepagents_api._final_report_payload(
        {
            "investigation_status": "resolved",
            "verification_status": "verified",
            "root_cause": "Verified cause",
        },
        "Why did this fail?",
    )
    assert payload["investigation_status"] == "resolved"
    assert payload["root_cause"] == "Verified cause"


def test_missing_stream_report_returns_terminal_diagnostic_not_500():
    payload = deepagents_api._final_report_payload(
        None, "Why did this fail?"
    )
    assert payload["investigation_status"] == "insufficient_evidence"
    assert payload["issue_summary"] == "Why did this fail?"


def test_graph_limit_supports_all_bounded_evidence_rounds():
    # The longest valid investigation revisits collection/validation several
    # times and must not collide with LangGraph's default transition ceiling.
    assert deepagents_main.INVESTIGATION_RECURSION_LIMIT >= 40


def test_shared_reasoning_service_reuses_one_model_for_all_stages():
    model = object()
    service = ReasoningService(
        model=model,
        max_concurrency=2,
        timeout_seconds=5,
    )
    assert service.model is model


def test_evidence_agent_has_enforced_model_and_tool_budgets(monkeypatch):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(evidence_agent, "create_agent", fake_create_agent)
    evidence_agent.create_evidence_agent({EvidenceSource.CODEBASE})
    middleware_names = {
        item.__class__.__name__ for item in captured["middleware"]
    }
    assert middleware_names == {
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    }


def test_report_validator_redacts_customer_identifiers():
    value = redact_report_text(
        "User alice@example.com, record 695653deffb2f9d2eccdd6d5, "
        "execution 8d2e0f80-d3f6-4a81-a3ae-e4bd4ea3a87c"
    )
    assert "alice@example.com" not in value
    assert "695653deffb2f9d2eccdd6d5" not in value
    assert "8d2e0f80-d3f6-4a81-a3ae-e4bd4ea3a87c" not in value


def test_evidence_validation_preserves_collection_error():
    update = validate_evidence_node({
        "evidence": [], "evidence_collection_errors": ["database unavailable"]
    })
    assert "database unavailable" in update["failure_reason"]


def test_query_understanding_extracts_arbitrary_domain_constraints():
    update = extract_business_entities_node({
        "user_query": (
            "Explain why deployment_key=release-42 failed for "
            "owner_email=ops@example.com"
        )
    })
    constraints = {
        item.field_hint: item.value
        for item in update["query_understanding"].constraints
    }
    assert constraints["deployment_key"] == "release-42"
    assert constraints["owner_email"] == "ops@example.com"
    assert update["query_understanding"].intent == "incident_investigation"


def test_every_project_query_plans_connected_code_and_database_sources(
    monkeypatch,
):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    understanding = extract_business_entities_node({
        "user_query": "How is invoice status calculated?"
    })["query_understanding"]
    update = plan_evidence_sources_node({
        "user_query": "How is invoice status calculated?",
        "query_understanding": understanding,
        "database_sources": [{"connection_id": "db-org-a"}],
        "codebase_sources": [{"connection_id": "code-org-a"}],
    })
    assert update["evidence_source_plan"].sources == [
        EvidenceSource.DATABASE,
        EvidenceSource.CODEBASE,
    ]
    assert {step.stage for step in update["investigation_plan"]} >= {
        "codebase", "schema", "database", "validation",
    }


def test_evidence_validation_updates_general_research_coverage():
    plan = [
        InvestigationPlanStep(
            stage="codebase",
            objective="Trace behavior",
            completion_criteria="Focused code exists",
        ),
        InvestigationPlanStep(
            stage="schema",
            objective="Find schema",
            completion_criteria="Schema exists",
        ),
        InvestigationPlanStep(
            stage="database",
            objective="Verify state",
            completion_criteria="Query succeeds",
        ),
    ]
    items = [
        Evidence(
            id="ev-code-general",
            evidence_type=EvidenceType.CODE_REFERENCE,
            source="service.py",
            summary="Focused symbol",
            content={"snippets": [{"path": "service.py", "content": "x"}]},
        ),
        Evidence(
            id="ev-schema-general",
            evidence_type=EvidenceType.DATABASE_SCHEMA,
            source="orders",
            summary="Schema",
            content={"schema": {"name": "orders"}},
        ),
        Evidence(
            id="ev-state-general",
            evidence_type=EvidenceType.DATABASE_QUERY,
            source="orders",
            summary="State",
            content={"rows": [], "row_count": 0, "error": None},
        ),
    ]
    update = validate_evidence_node({
        "evidence": items,
        "investigation_plan": plan,
    })
    assert all(
        step.status == "completed"
        for step in update["investigation_plan"]
    )


def test_root_cause_rejects_hallucinated_references():
    analysis = RootCauseAnalysis(root_cause="cause", confidence=.95, is_established=True,
        selected_hypothesis_id="made-up", supporting_evidence_ids=["made-up"],
        suggested_actions=["Delete and recreate the customer organization."])
    update = validate_root_cause_node({"evidence": [evidence()], "investigation": investigation(),
                                       "root_cause_analysis": analysis})
    assert update["root_cause_analysis"].is_established is False
    assert update["root_cause_analysis"].root_cause is None
    assert update["root_cause_analysis"].supporting_evidence_ids == []
    assert update["root_cause_analysis"].confidence == 0.0
    assert update["root_cause_analysis"].suggested_actions == []


def test_root_cause_removes_direct_data_mutation_and_speculation():
    causal_evidence = evidence().model_copy(deep=True)
    causal_evidence.content["evidence_role"] = "causal_validation"
    analysis = RootCauseAnalysis(
        root_cause="The verified feature predicate rejected the operation.",
        confidence=.9,
        is_established=True,
        selected_hypothesis_id="h-1",
        supporting_evidence_ids=["ev-1"],
        recommended_fix=[
            "Manually create or repair the subscription record in MongoDB.",
            "Reconcile the subscription through the supported billing workflow.",
        ],
        contributing_factors=[
            "The absence suggests a potential provisioning failure.",
            "The billing event is confirmed to be missing.",
        ],
    )
    update = validate_root_cause_node({
        "evidence": [causal_evidence],
        "investigation": investigation(),
        "root_cause_analysis": analysis,
    })
    validated = update["root_cause_analysis"]
    assert validated.is_established is True
    assert validated.recommended_fix == [
        "Reconcile the subscription through the supported billing workflow."
    ]
    assert validated.contributing_factors == [
        "The billing event is confirmed to be missing."
    ]


def test_exploratory_lookup_cannot_become_customer_root_cause():
    exploratory = Evidence(
        id="ev-lookup",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="organisations",
        summary="Exploratory organization lookup returned no rows",
        content={
            "evidence_role": "exploration",
            "filter": {"name": "bigbrosai"},
            "rows": [],
            "row_count": 0,
            "error": None,
        },
    )
    result = investigation()
    result.hypotheses[0].supporting_evidence_ids = ["ev-lookup"]
    analysis = RootCauseAnalysis(
        root_cause=(
            "The operation was rejected because TraceX's case-sensitive "
            "organization lookup returned no rows."
        ),
        confidence=.95,
        is_established=True,
        selected_hypothesis_id="h-1",
        supporting_evidence_ids=["ev-lookup"],
    )
    update = validate_root_cause_node({
        "evidence": [exploratory],
        "investigation": result,
        "root_cause_analysis": analysis,
    })
    assert update["root_cause_analysis"].is_established is False
    assert update["root_cause_analysis"].root_cause is None
    assert update["root_cause_analysis"].supporting_evidence_ids == []
    assert update["root_cause_analysis"].confidence == 0.0


def test_external_web_evidence_cannot_independently_establish_root_cause():
    external = Evidence(
        id="ev-web",
        evidence_type=EvidenceType.API_RESPONSE,
        source="https://docs.example.com/error",
        summary="Official documentation describes an error",
        content={
            "external_context_only": True,
            "url": "https://docs.example.com/error",
        },
    )
    analysis = RootCauseAnalysis(
        root_cause="The customer incident was caused by the documented error.",
        confidence=.9,
        is_established=True,
        selected_hypothesis_id="h-1",
        supporting_evidence_ids=["ev-web"],
    )
    update = validate_root_cause_node({
        "evidence": [external],
        "investigation": investigation(),
        "root_cause_analysis": analysis,
    })
    assert update["root_cause_analysis"].is_established is False
    assert update["root_cause_analysis"].root_cause is None


def test_compact_evidence_deduplicates_repeated_operations_and_bounds_rows():
    repeated = [
        Evidence(
            id=f"ev-{index}",
            evidence_type=EvidenceType.DATABASE_QUERY,
            source="mongo-1",
            summary="MongoDB query returned 20 documents",
            content={
                "collection": "users",
                "filter": {"status": "ACTIVE"},
                "rows": [{"index": row} for row in range(20)],
            },
        )
        for index in range(3)
    ]
    payload = compact_evidence(repeated)
    assert len(payload) == 1
    assert payload[0]["id"] == "ev-2"
    assert len(payload[0]["content"]["rows"]) == 10


def test_evidence_collection_persists_customer_report_as_evidence(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, *_args, **_kwargs):
            return {}

    investigation_id = "customer-report-test"
    monkeypatch.setattr(
        evidence_collection,
        "create_evidence_agent",
        lambda _sources: FakeAgent(),
    )

    async def scenario():
        await evidence_repository.clear(investigation_id)
        update = await evidence_collection.collect_evidence_node({
            "investigation_id": investigation_id,
            "user_query": (
                "Campaign creation shows: Your current plan does not include "
                "this campaign type."
            ),
            "organization_id": "org-control-plane",
            "project_id": "project-control-plane",
            "database_sources": [],
            "evidence_collection_attempts": 0,
        })
        reports = [
            item for item in update["evidence"]
            if item.evidence_type == EvidenceType.USER_INPUT
        ]
        assert len(reports) == 1
        assert "current plan" in reports[0].content["reported_text"]
        await evidence_repository.clear(investigation_id)

    asyncio.run(scenario())


def test_evidence_collection_runs_planned_sources_as_isolated_workers(monkeypatch):
    created = []

    class FakeAgent:
        def __init__(self, sources):
            self.sources = sources

        async def ainvoke(self, *_args, **_kwargs):
            await asyncio.sleep(0)
            return {}

    def create(sources):
        created.append({item.value for item in sources})
        return FakeAgent(sources)

    monkeypatch.setattr(evidence_collection, "create_evidence_agent", create)
    investigation_id = "parallel-source-test"

    async def scenario():
        await evidence_repository.clear(investigation_id)
        await evidence_collection.collect_evidence_node({
            "investigation_id": investigation_id,
            "user_query": "Why did the operation fail?",
            "organization_id": "org-1",
            "project_id": "project-1",
            "database_sources": [],
            "evidence_source_plan": EvidenceSourcePlan(sources=[
                EvidenceSource.DATABASE, EvidenceSource.LOGS,
            ]),
            "evidence_collection_attempts": 0,
        })
        await evidence_repository.clear(investigation_id)

    asyncio.run(scenario())
    assert created == [{"database"}, {"logs"}]


def test_sqlite_checkpoint_provider_persists_thread_state(monkeypatch, tmp_path):
    from langgraph.graph import END, START, StateGraph

    monkeypatch.setenv(
        "CHECKPOINT_DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'checkpoints.sqlite'}",
    )

    async def scenario():
        async with checkpoint_provider() as saver:
            builder = StateGraph(dict)
            builder.add_node("increment", lambda state: {"value": state["value"] + 1})
            builder.add_edge(START, "increment")
            builder.add_edge("increment", END)
            graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "durable-thread"}}
            assert (await graph.ainvoke({"value": 1}, config))["value"] == 2
        async with checkpoint_provider() as saver:
            builder = StateGraph(dict)
            builder.add_node("increment", lambda state: state)
            builder.add_edge(START, "increment")
            builder.add_edge("increment", END)
            graph = builder.compile(checkpointer=saver)
            snapshot = await graph.aget_state(
                {"configurable": {"thread_id": "durable-thread"}}
            )
            assert snapshot.values["value"] == 2

    asyncio.run(scenario())


def test_unsupported_native_structured_output_repairs_json(monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

    class BrokenStructured:
        async def ainvoke(self, _messages):
            raise ValueError("response_format is unsupported by this model")

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, _schema):
            return BrokenStructured()

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return Response('{"root_cause": 12}')
            return Response("""```json
            {
              "root_cause": null,
              "confidence": 0,
              "is_established": false,
              "reasoning_summary": "Insufficient evidence"
            }
            ```""")

    async def scenario():
        result = await invoke_structured(
            FakeModel(),
            RootCauseAnalysis,
            [{"role": "user", "content": "Analyze"}],
            stage="test",
        )
        assert result.is_established is False
        assert result.reasoning_summary == "Insufficient evidence"

    asyncio.run(scenario())


def test_failed_column_assumption_is_recorded_for_correction():
    failed = Evidence(
        id="ev-failed-column",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="db-1",
        summary="Read-only query failed: UndefinedColumn",
        content={
            "query": "SELECT payment_status FROM payments",
            "rows": [],
            "row_count": 0,
            "error": 'column "payment_status" does not exist',
            "evidence_role": "exploration",
        },
    )
    update = self_check_node({"evidence": [failed]})
    assert update["failed_assumptions"]
    assert update["failed_assumptions"][0].retryable is True
    assert "Re-inspect schema" in update["failed_assumptions"][0].correction


def test_empty_final_result_requests_semantic_value_discovery():
    empty = Evidence(
        id="ev-empty",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="db-1",
        summary="No successful payments",
        content={
            "query": "SELECT * FROM payments WHERE status_code = 'Success'",
            "rows": [],
            "row_count": 0,
            "error": None,
            "evidence_role": "final_answer",
        },
    )
    update = self_check_node({"evidence": [empty]})
    assert any(
        "stored values" in request
        for request in update["requested_evidence"]
    )


def test_encoded_status_values_are_discovered_with_native_mongodb_pipeline(
    monkeypatch,
):
    captured = {}

    def fake_mongodb_query(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "mongodb",
            "rows": [
                {"value": "S", "count": 5},
                {"value": "F", "count": 2},
                {"value": "P", "count": 1},
            ],
            "row_count": 3,
            "error": None,
        }

    monkeypatch.setattr(
        evidence_tools,
        "_run_mongodb_query",
        fake_mongodb_query,
    )
    database_token = bind_database_sources([{
        "connection_id": "mongo-1",
        "provider": "mongodb",
        "connection_url": "mongodb://example.invalid/test",
        "analysis": {
            "objects": [{
                "namespace": "db",
                "name": "payments",
                "fields": [{
                    "name": "status_code",
                    "data_type": "str",
                }],
            }],
        },
    }])
    investigation_token = bind_investigation("encoded-status-test")
    try:
        result = asyncio.run(evidence_tools.discover_field_values(
            "payments",
            "status_code",
        ))
        assert [row["value"] for row in result["rows"]] == ["S", "F", "P"]
        assert captured["pipeline"][0] == {
            "$group": {
                "_id": "$status_code",
                "count": {"$sum": 1},
            }
        }
    finally:
        reset_investigation(investigation_token)
        reset_database_sources(database_token)
        asyncio.run(evidence_repository.clear("encoded-status-test"))


def test_contradictory_code_and_database_evidence_remains_inconclusive():
    code = Evidence(
        id="ev-code",
        evidence_type=EvidenceType.CODE_REFERENCE,
        source="payment.service.py:mark_success",
        summary="Code maps successful callback to status S",
        content={"status": "S"},
    )
    database = Evidence(
        id="ev-db",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="payments",
        summary="Payment remains pending",
        content={
            "rows": [{"status_code": "P"}],
            "row_count": 1,
            "error": None,
            "evidence_role": "exploration",
        },
    )
    result = investigation().model_copy(update={
        "actual_state": "Code expects S while the stored record remains P.",
        "requires_more_evidence": True,
        "requested_evidence": ["Retrieve callback/job logs for the payment."],
    })
    assert route_after_investigation({
        "user_query": "Why is this successful payment pending?",
        "evidence": [code, database],
        "investigation": result,
        "evidence_collection_attempts": 1,
    }) == "collect_more_evidence"


def test_report_validator_rejects_root_cause_without_evidence():
    report = build_final_report_node({
        "user_query": "Why did payment fail?",
        "evidence": [evidence()],
        "investigation": investigation(),
        "root_cause_analysis": RootCauseAnalysis(
            root_cause="Unsupported cause",
            confidence=0.95,
            is_established=True,
            supporting_evidence_ids=["invented"],
        ),
    })["final_report"]
    report.root_cause = "Unsupported cause"
    report.supporting_evidence_ids = ["invented"]
    update = validate_report_node({
        "evidence": [evidence()],
        "final_report": report,
    })
    assert update["final_report"].root_cause is None
    assert update["insufficient_evidence"] is True
    assert "Unsupported cause" not in update["final_report"].customer_response
    assert "reliable root cause" in update["final_report"].customer_response


def test_report_validator_rejects_exploration_as_customer_rca():
    exploratory = Evidence(
        id="ev-lookup",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="organisations",
        summary="Case-sensitive exploration returned no rows",
        content={
            "evidence_role": "exploration",
            "rows": [],
            "row_count": 0,
            "error": None,
        },
    )
    analysis = RootCauseAnalysis(
        root_cause="TraceX's lowercase lookup caused the payment failure.",
        confidence=.95,
        is_established=True,
        supporting_evidence_ids=["ev-lookup"],
        recommended_fix=["Change the customer application search."],
    )
    report = build_final_report_node({
        "user_query": "Why did payment fail?",
        "evidence": [exploratory],
        "investigation": investigation(),
        "root_cause_analysis": analysis,
    })["final_report"]
    update = validate_report_node({
        "evidence": [exploratory],
        "root_cause_analysis": analysis,
        "final_report": report,
    })
    validated = update["final_report"]
    assert validated.root_cause is None
    assert validated.recommended_fix == []
    assert "lowercase lookup" not in validated.customer_response
    assert "lowercase lookup" not in validated.engineering_note
    assert validated.verification_status == "inconclusive"


def test_entitlement_causal_chain_produces_actionable_l2_report():
    evidence_items = [
        Evidence(
            id="ev-caller",
            evidence_type=EvidenceType.CODE_REFERENCE,
            source="campaign.service.ts",
            summary="Campaign creation validates apiCampaign",
            content={
                "path": "src/campaign/services/campaign.service.ts",
                "symbol": "validateSubscriptionFeature",
                "content": (
                    "validateSubscriptionFeature(projectId, "
                    "SUBSCRIPTION_FEATURES_ENUM.apiCampaign)"
                ),
            },
        ),
        Evidence(
            id="ev-validator",
            evidence_type=EvidenceType.CODE_REFERENCE,
            source="subscription.service.ts",
            summary="Validator selects accessible unexpired subscription",
            content={
                "collection": "subscriptions",
                "filter": {
                    "projectId": "resolved project id",
                    "status": {"$in": ["ACTIVE"]},
                    "currentTermEnd": {"$gte": "now"},
                },
            },
        ),
        Evidence(
            id="ev-state",
            evidence_type=EvidenceType.DATABASE_QUERY,
            source="mongodb",
            summary="Project has active FREE and pending PRO subscriptions",
            content={
                "collection": "subscriptions",
                "rows": [
                    {
                        "status": "ACTIVE",
                        "planSnapshot": {"code": "FREE"},
                        "currentTermEnd": "future",
                    },
                    {
                        "status": "PENDING",
                        "planSnapshot": {"code": "PRO"},
                        "currentTermEnd": None,
                    },
                ],
                "row_count": 2,
                "evidence_role": "exploration",
            },
        ),
        Evidence(
            id="ev-mapping",
            evidence_type=EvidenceType.CODE_REFERENCE,
            source="subscription-features.constant.ts",
            summary="FREE disables apiCampaign while BASIC and PRO enable it",
            content={
                "FREE": {"apiCampaign": False},
                "BASIC": {"apiCampaign": True},
                "PRO": {"apiCampaign": True},
            },
        ),
    ]
    result = InvestigationResult(
        issue_summary="Campaign creation is rejected for the project.",
        expected_state="Campaign creation succeeds on an eligible paid plan.",
        actual_state=(
            "The validator selects the active FREE subscription; pending PRO "
            "records are ineligible, and FREE disables apiCampaign."
        ),
        affected_components=["CampaignService", "SubscriptionService"],
        hypotheses=[
            Hypothesis(
                id="h-entitlement",
                statement=(
                    "The active FREE subscription causes the apiCampaign "
                    "feature gate to reject campaign creation."
                ),
                supporting_evidence_ids=[
                    "ev-caller", "ev-validator", "ev-state", "ev-mapping",
                ],
                confidence=0.98,
                status="supported",
            )
        ],
    )
    analysis = RootCauseAnalysis(
        root_cause=(
            "Campaign creation was rejected because the active plan is FREE, "
            "which causes the apiCampaign entitlement check to evaluate false."
        ),
        confidence=0.98,
        selected_hypothesis_id="h-entitlement",
        supporting_evidence_ids=[
            "ev-caller", "ev-validator", "ev-state", "ev-mapping",
        ],
        reasoning_summary=(
            "The entry point, selection predicate, stored subscription state, "
            "and feature mapping form a complete causal chain."
        ),
        is_established=True,
        contributing_factors=["Paid upgrade records remain pending."],
        recommended_fix=[
            "Complete or reconcile the paid-plan activation through billing.",
            "Cancel superseded pending upgrade records.",
        ],
        validation_steps=[
            "Confirm one eligible paid subscription is active and unexpired.",
            "Retry campaign creation.",
        ],
    )
    validated = validate_root_cause_node({
        "evidence": evidence_items,
        "investigation": result,
        "root_cause_analysis": analysis,
    })
    assert validated["root_cause_analysis"].is_established is True
    report = build_final_report_node({
        "user_query": "Campaign creation shows upgrade required.",
        "evidence": evidence_items,
        "investigation": result,
        "root_cause_analysis": validated["root_cause_analysis"],
    })["final_report"]
    assert report.verification_status == "verified"
    assert report.investigation_status == "resolved"
    assert "active plan is FREE" in report.root_cause
    assert "Recommended resolution" in report.customer_response
    assert "reconcile the paid-plan activation" in report.customer_response
    assert report.contributing_factors == [
        "Paid upgrade records remain pending."
    ]


def test_verified_cross_source_explanation_is_not_marked_inconclusive():
    understanding = extract_business_entities_node({
        "user_query": "How is invoice status calculated?"
    })["query_understanding"]
    result = InvestigationResult(
        issue_summary="Invoice status calculation",
        expected_state="The status follows the documented state transition.",
        actual_state=(
            "InvoiceService maps paidAt to PAID, and the stored invoice has a "
            "non-null paidAt value, so its status is PAID."
        ),
        affected_components=["InvoiceService", "invoices"],
        hypotheses=[],
        unresolved_questions=[],
        requires_more_evidence=False,
        requested_evidence=[],
    )
    items = [
        Evidence(
            id="ev-code-explanation",
            evidence_type=EvidenceType.CODE_REFERENCE,
            source="invoice.service.ts",
            summary="Invoice status mapping",
            content={"snippets": [{"content": "paidAt ? PAID : PENDING"}]},
        ),
        Evidence(
            id="ev-db-explanation",
            evidence_type=EvidenceType.DATABASE_QUERY,
            source="invoices",
            summary="Stored invoice state",
            content={
                "rows": [{"paidAt": "2026-07-25T00:00:00Z"}],
                "row_count": 1,
                "error": None,
                "evidence_role": "exploration",
            },
        ),
    ]
    plan = [
        InvestigationPlanStep(
            stage="codebase", objective="Trace mapping", status="completed",
            evidence_ids=["ev-code-explanation"],
        ),
        InvestigationPlanStep(
            stage="database", objective="Verify state", status="completed",
            evidence_ids=["ev-db-explanation"],
        ),
        InvestigationPlanStep(stage="validation", objective="Cross-check"),
    ]
    report = build_final_report_node({
        "user_query": "How is invoice status calculated?",
        "query_understanding": understanding,
        "investigation_plan": plan,
        "evidence": items,
        "investigation": result,
        "root_cause_analysis": RootCauseAnalysis(
            is_established=False,
            reasoning_summary="This is an explanation, not an incident.",
        ),
    })["final_report"]
    assert report.response_type == "explanation"
    assert report.verification_status == "verified"
    assert report.investigation_status == "resolved"
    assert "Verified findings" in report.customer_response


def test_report_builder_flattens_query_rows():
    update = build_final_report_node({"user_query": "get row", "evidence": [evidence()],
        "investigation": investigation(), "root_cause_analysis": RootCauseAnalysis()})
    assert update["final_report"].result_records[0].record == {"id": 1}


def test_retrieval_report_excludes_exploratory_query_rows():
    exploratory = Evidence(
        id="ev-status", evidence_type=EvidenceType.DATABASE_QUERY,
        source="db", summary="statuses", content={
            "rows": [{"status": "S"}],
            "evidence_role": "exploration",
            "error": None,
        },
    )
    final = Evidence(
        id="ev-bookings", evidence_type=EvidenceType.DATABASE_QUERY,
        source="db", summary="bookings",
        content={
            "rows": [{"bookingid": "1"}, {"bookingid": "2"}],
            "evidence_role": "final_answer",
            "error": None,
        },
    )
    report = build_final_report_node({
        "user_query": "give bookings", "evidence": [exploratory, final],
        "investigation": investigation(), "root_cause_analysis": None,
    })["final_report"]
    assert [item.record for item in report.result_records] == [
        {"bookingid": "1"}, {"bookingid": "2"}
    ]
    assert report.supporting_evidence_ids == ["ev-bookings"]
    assert report.missing_information == []


def test_retrieval_with_rows_skips_root_cause_model():
    payment = evidence().model_copy(deep=True)
    payment.content["rows"] = [{"paymentId": "pay-1", "status": "SUCCESS"}]
    state = {
        "user_query": "Give me successful payments",
        "evidence": [payment],
        "investigation": investigation(),
    }
    assert route_after_investigation(state) == "build_result_report"


def test_unverified_retrieval_collects_more_evidence_then_stops():
    unresolved = investigation().model_copy(update={
        "actual_state": "The user relationship is not explicitly defined.",
        "unresolved_questions": ["Which user owns this wallet?"],
    })
    state = {
        "user_query": "Give me the user with the highest wallet balance",
        "evidence": [evidence()],
        "investigation": unresolved,
        "evidence_collection_attempts": 1,
    }
    assert route_after_investigation(state) == "collect_more_evidence"
    state["evidence_collection_attempts"] = 3
    assert route_after_investigation(state) == "build_inconclusive_report"


def test_verified_retrieval_has_customer_ready_response():
    payment = evidence().model_copy(deep=True)
    payment.content["rows"] = [{"paymentId": "pay-1", "status": "SUCCESS"}]
    report = build_final_report_node({
        "user_query": "Give me one payment",
        "evidence": [payment],
        "investigation": investigation(),
        "root_cause_analysis": None,
    })["final_report"]
    assert report.customer_response.startswith(
        "Subject: TraceX investigation update\n\nHello,\n\n"
    )
    assert "verified the result" in report.customer_response
    assert "Status: SUCCESS" in report.customer_response
    assert "Payment Id" not in report.customer_response
    assert report.customer_response.endswith("Regards,\nTraceX L2 Support Team")
    assert report.result_records
    assert report.response_type == "retrieval"
    assert report.verification_status == "verified"


def test_unverified_retrieval_does_not_expose_exploratory_rows():
    unresolved = investigation().model_copy(update={
        "actual_state": "The relationship is not explicitly defined.",
        "unresolved_questions": ["Relationship evidence is missing."],
    })
    report = build_final_report_node({
        "user_query": "Find the user with the highest wallet balance",
        "evidence": [evidence()],
        "investigation": unresolved,
        "root_cause_analysis": None,
    })["final_report"]
    assert report.verification_status == "inconclusive"
    assert report.result_records == []
    assert "not sufficient" in report.customer_response


def test_final_answer_evidence_wins_over_later_fallback_scan():
    final = Evidence(
        id="ev-final",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="atomic answer",
        content={
            "evidence_role": "final_answer",
            "error": None,
            "pipeline": [{"$sort": {"balance": -1}}, {"$limit": 1}],
            "rows": [{
                "organizationName": "BIGBROS AI",
                "organizationId": "org-1",
                "walletBalance": 100432090.5,
                "currency": "INR",
                "walletStatus": "ACTIVE",
            }],
        },
    )
    fallback = Evidence(
        id="ev-fallback",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="fallback scan",
        content={
            "evidence_role": "exploration",
            "error": None,
            "rows": [{"name": f"Organization {index}"} for index in range(10)],
        },
    )
    report = build_final_report_node({
        "user_query": "Find the organization with the highest wallet balance",
        "evidence": [final, fallback],
        "investigation": investigation(),
        "root_cause_analysis": None,
    })["final_report"]
    assert len(report.result_records) == 1
    assert report.result_records[0].record["organizationName"] == "BIGBROS AI"
    assert "BIGBROS AI" in report.customer_response
    assert "Wallet Balance: 100,432,090.5" in report.customer_response
    assert "Organization Id" not in report.customer_response
    assert "10 matching" not in report.customer_response


def test_customer_response_omits_sensitive_fields_and_keeps_safe_metrics():
    final = Evidence(
        id="ev-sensitive",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="verified organization balance",
        content={
            "evidence_role": "final_answer",
            "error": None,
            "rows": [{
                "_id": "695653deffb2f9d2eccdd6d5",
                "orgId": "695653deffb2f9d2eccdd6d1",
                "orgName": "BIGBROS AI",
                "balance": 100432090.5,
                "paidAmount": 2500,
                "email": "finance@example.com",
                "phone": "+91-9999999999",
                "apiToken": "top-secret-token",
                "billingAddress": "restricted",
            }],
        },
    )
    report = build_final_report_node({
        "user_query": "Identify the organization with the highest balance",
        "evidence": [final],
        "investigation": investigation(),
        "root_cause_analysis": None,
    })["final_report"]
    response = report.customer_response
    assert "Org Name: BIGBROS AI" in response
    assert "Balance: 100,432,090.5" in response
    assert "Paid Amount: 2,500" in response
    for sensitive_value in (
        "695653deffb2f9d2eccdd6d5",
        "695653deffb2f9d2eccdd6d1",
        "finance@example.com",
        "+91-9999999999",
        "top-secret-token",
        "restricted",
    ):
        assert sensitive_value not in response


@pytest.mark.parametrize(
    ("provider", "expected_tool", "excluded_tool"),
    [
        ("mongodb", "run_safe_mongodb_query", "run_safe_read_query"),
        ("postgresql", "run_safe_read_query", "run_safe_mongodb_query"),
        ("mysql", "run_safe_read_query", "run_safe_mongodb_query"),
        ("oracle", "run_safe_read_query", "run_safe_mongodb_query"),
    ],
)
def test_evidence_agent_exposes_only_provider_compatible_query_tool(
    monkeypatch, provider, expected_tool, excluded_tool
):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(evidence_agent, "create_agent", fake_create_agent)
    token = bind_database_sources([{
        "connection_id": "db-1",
        "provider": provider,
        "connection_url": "redacted",
        "analysis": {"database_type": provider, "objects": []},
    }])
    try:
        evidence_agent.create_evidence_agent({EvidenceSource.DATABASE})
    finally:
        reset_database_sources(token)

    tool_names = {
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in captured["tools"]
    }
    assert expected_tool in tool_names
    assert excluded_tool not in tool_names
    assert f"db-1={provider}" in captured["system_prompt"]


def test_incident_report_does_not_publish_exploratory_database_rows():
    user_record = Evidence(
        id="ev-user",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="user lookup",
        content={
            "evidence_role": "exploration",
            "error": None,
            "rows": [{
                "email": "person@example.com",
                "password": "hashed-secret",
                "phone": "+91-9999999999",
                "status": "ACTIVE",
            }],
        },
    )
    report = build_final_report_node({
        "user_query": "Why can person@example.com not create a campaign?",
        "evidence": [user_record, user_record],
        "investigation": investigation(),
        "root_cause_analysis": None,
    })["final_report"]
    assert report.response_type == "incident"
    assert report.result_records == []


def test_cancel_endpoint_stops_registered_investigation():
    async def scenario():
        task = asyncio.create_task(asyncio.sleep(60))
        deepagents_api._active_investigations["execution-1"] = (
            task,
            "internal-secret",
        )
        try:
            response = await deepagents_api.cancel_investigation(
                "execution-1",
                "internal-secret",
            )
            assert response == {"status": "cancellation_requested"}
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            deepagents_api._active_investigations.pop("execution-1", None)

    asyncio.run(scenario())


def test_failed_final_query_is_never_customer_result():
    failed = Evidence(
        id="ev-failed",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="query failed",
        content={
            "evidence_role": "final_answer",
            "error": "invalid ObjectId",
            "rows": [],
        },
    )
    report = build_final_report_node({
        "user_query": "Find the organization with the highest wallet balance",
        "evidence": [failed],
        "investigation": investigation(),
        "root_cause_analysis": None,
    })["final_report"]
    assert report.verification_status == "inconclusive"
    assert report.result_records == []


def test_failed_query_is_persisted_for_diagnostics(monkeypatch):
    def fail_query(**kwargs):
        raise ValueError("invalid ObjectId")

    monkeypatch.setattr(evidence_tools, "_run_mongodb_query", fail_query)
    async def exercise():
        token = bind_investigation("inv-query-failure")
        try:
            with pytest.raises(ValueError):
                await evidence_tools.run_safe_mongodb_query(
                    collection="organisations",
                    filter_query={"_id": "invalid"},
                    purpose="final_answer",
                )
            saved = await evidence_repository.list_by_investigation(
                "inv-query-failure"
            )
            assert len(saved) == 1
            assert saved[0].content["evidence_role"] == "final_answer"
            assert saved[0].content["rows"] == []
            assert saved[0].content["error"] == "invalid ObjectId"
        finally:
            reset_investigation(token)
            await evidence_repository.clear("inv-query-failure")

    asyncio.run(exercise())


def test_sql_validation_rejects_writes_and_multiple_statements():
    validate_read_only_query("SELECT * FROM payments")
    assert apply_limit("SELECT 1") == "SELECT 1 LIMIT 100;"
    for query in (
        "DELETE FROM payments",
        "SELECT 1; DROP TABLE payments",
        "CALL dangerous_procedure()",
    ):
        try:
            validate_read_only_query(query)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe SQL accepted")


def test_provider_specific_sql_limits():
    assert apply_limit("SELECT * FROM users", "mysql") == (
        "SELECT * FROM users LIMIT 100;"
    )
    assert apply_limit("SELECT * FROM users", "oracle") == (
        "SELECT * FROM users FETCH FIRST 100 ROWS ONLY"
    )
    assert apply_limit("SELECT * FROM users LIMIT 500", "postgresql") == (
        "SELECT * FROM users LIMIT 100;"
    )
    assert apply_limit(
        "SELECT * FROM users FETCH FIRST 500 ROWS ONLY",
        "oracle",
    ) == "SELECT * FROM users FETCH FIRST 100 ROWS ONLY"


def test_mongodb_sort_directions_accept_common_agent_values():
    assert _normalize_sort_direction("desc") == -1
    assert _normalize_sort_direction("descending") == -1
    assert _normalize_sort_direction("-1") == -1
    assert _normalize_sort_direction(-1) == -1
    assert _normalize_sort_direction("asc") == 1
    assert _normalize_sort_direction("ascending") == 1
    assert _normalize_sort_direction("1") == 1
    assert _normalize_sort_direction(1) == 1
    assert _normalize_mongodb_sort({"balance": "desc"}) == [("balance", -1)]
    assert _normalize_mongodb_sort([["createdAt", "asc"]]) == [
        ("createdAt", 1)
    ]
    try:
        _normalize_mongodb_sort({"balance": "sideways"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid MongoDB sort direction accepted")


def test_mongodb_pipeline_combines_filter_and_caps_limit():
    pipeline = _bounded_mongodb_pipeline(
        [{"$sort": {"balance": -1}}, {"$limit": 500}],
        {"status": "ACTIVE"},
        100,
    )
    assert pipeline == [
        {"$match": {"status": "ACTIVE"}},
        {"$sort": {"balance": -1}},
        {"$limit": 100},
    ]


def test_mongodb_nested_id_shape_is_normalized():
    value = _coerce_mongodb_ids({
        "_id": {"_id": "695653deffb2f9d2eccdd6d1"}
    })
    assert str(value["_id"]) == "695653deffb2f9d2eccdd6d1"
    alias = _coerce_mongodb_ids({
        "_id": {"_oid": "695653deffb2f9d2eccdd6d1"}
    })
    assert str(alias["_id"]) == "695653deffb2f9d2eccdd6d1"


def test_mongodb_rejects_scientific_notation_for_large_identifiers():
    with pytest.raises(ValueError, match="exact string"):
        _coerce_mongodb_ids({
            "organisationId": 6.956536868588889e23,
        })


def test_relational_query_routes_to_each_provider(monkeypatch):
    monkeypatch.setattr(
        database_tools,
        "_query_postgresql",
        lambda url, query: {"provider": "postgresql", "query": query},
    )
    monkeypatch.setattr(
        database_tools,
        "_query_mysql",
        lambda url, query: {"provider": "mysql", "query": query},
    )
    monkeypatch.setattr(
        database_tools,
        "_query_oracle",
        lambda url, query: {"provider": "oracle", "query": query},
    )
    for provider in ("postgresql", "mysql", "oracle"):
        token = bind_database_sources([{
            "connection_id": f"{provider}-1",
            "provider": provider,
            "connection_url": f"{provider}://example.invalid/database",
            "analysis": {"objects": []},
        }])
        try:
            result = database_tools.run_safe_read_query("SELECT 1")
            assert result["provider"] == provider
        finally:
            reset_database_sources(token)


def test_mongodb_validation_rejects_write_and_server_code_stages():
    for operation in (
        [{"$match": {}}, {"$out": "stolen"}],
        {"$where": "function () { return true; }"},
        [{"$project": {"value": {"$function": {"body": "x"}}}}],
    ):
        try:
            _validate_mongodb_value(operation)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe MongoDB operation accepted")


def test_mongodb_validation_rejects_quoted_operator_keys():
    with pytest.raises(ValueError, match="Malformed quoted MongoDB operator"):
        _validate_mongodb_value({
            "organisationId": {
                "'$eq'": {"'$oid'": "695653deffb2f9d2eccdd6d1"}
            }
        })


def test_client_queries_reject_tracex_control_plane_ids():
    token = bind_database_sources([], {"org-control-id", "project-control-id"})
    try:
        _reject_control_plane_values({
            "$match": {"organisationId": "org-control-id"}
        })
    except ValueError:
        pass
    else:
        raise AssertionError("TraceX organization ID accepted as client filter")
    finally:
        reset_database_sources(token)


def test_rate_limit_retry_delay_is_extracted():
    error = RuntimeError("429 RESOURCE_EXHAUSTED; Please retry in 52.063s")
    assert is_rate_limit_error(error)
    assert retry_delay_seconds(error) == 52.063


def test_vectorless_schema_retrieval_does_not_invent_domain_synonyms():
    token = bind_database_sources([{
        "connection_id": "db-1",
        "provider": "postgresql",
        "connection_url": "postgresql://example.invalid/test",
        "analysis": {
            "objects": [
                {"namespace": "public", "name": "payment", "fields": []},
                {"namespace": "public", "name": "orders", "fields": []},
            ]
        },
    }])
    try:
        matches = retrieve_schema_context("bookings with successful payments", limit=8)
        names = {item["name"] for item in matches}
        assert "payment" in names
        assert "orders" not in names
    finally:
        reset_database_sources(token)


def test_vectorless_retrieval_expands_direct_relationships():
    token = bind_database_sources([{
        "connection_id": "mongo-1",
        "provider": "mongodb",
        "connection_url": "mongodb://example.invalid/test",
        "analysis": {
            "objects": [
                {"namespace": "db", "name": "wallets", "fields": [
                    {"name": "organisationId", "data_type": "ObjectId"}
                ]},
                {"namespace": "db", "name": "organisations", "fields": [
                    {"name": "_id", "data_type": "ObjectId"}
                ]},
            ],
            "relationships": [{
                "source_object": "wallets",
                "source_field": "organisationId",
                "target_object": "organisations",
                "target_field": "_id",
                "origin": "inferred",
                "confidence": 0.9,
            }],
        },
    }])
    try:
        matches = retrieve_schema_context("highest wallet balance", limit=8)
        assert [item["name"] for item in matches] == [
            "wallets",
            "organisations",
        ]
        assert matches[1]["retrieval_reason"] == "Related to wallets"
    finally:
        reset_database_sources(token)


def test_query_understanding_extracts_business_identifiers():
    update = extract_business_entities_node({
        "user_query": "Why did bookingid BK-1042 fail for user id U-9?"
    })
    understanding = update["query_understanding"]
    values = {item.entity_type: item.value for item in understanding.entities}
    assert values["booking_id"] == "BK-1042"
    assert values["user_id"] == "U-9"
    assert understanding.intent == "incident_investigation"


def test_identify_query_is_classified_as_data_retrieval():
    understanding = extract_business_entities_node({
        "user_query": "Identify the organization with the highest wallet balance"
    })["query_understanding"]
    assert understanding.intent == "data_retrieval"
    assert {item.entity_type for item in understanding.entities} >= {
        "organization",
        "wallet",
    }


def test_which_question_is_classified_as_data_retrieval():
    understanding = extract_business_entities_node({
        "user_query": "Which organization has run the highest number of campaigns?"
    })["query_understanding"]
    assert understanding.intent == "data_retrieval"


def test_pricing_question_is_classified_as_informational():
    understanding = extract_business_entities_node({
        "user_query": "What are the WhatsApp rates for North America?"
    })["query_understanding"]
    assert understanding.intent == "informational"


def test_question_containing_failure_is_classified_as_incident():
    understanding = extract_business_entities_node({
        "user_query": "What error caused campaign creation to fail?"
    })["query_understanding"]
    assert understanding.intent == "incident_investigation"


def test_informational_request_never_becomes_root_cause():
    understanding = extract_business_entities_node({
        "user_query": "What are the WhatsApp rates for North America?"
    })["query_understanding"]
    pricing = Evidence(
        id="ev-pricing",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="whatsapppricingrates",
        summary="Regional pricing",
        content={
            "evidence_role": "final_answer",
            "error": None,
            "row_count": 1,
            "rows": [{
                "region": "North America",
                "marketing": 2.1976,
                "utility": 0.2989,
                "authentication": 0.2989,
                "currency": "INR",
            }],
        },
    )
    official_documentation = Evidence(
        id="ev-pricing-docs",
        evidence_type=EvidenceType.API_RESPONSE,
        source="public_web",
        summary="Official pricing documentation",
        content={
            "external_context_only": True,
            "citations": [{
                "title": "Official WhatsApp pricing documentation",
                "url": "https://developers.facebook.com/docs/whatsapp/pricing",
            }],
        },
    )
    result = investigation().model_copy(update={
        "issue_summary": "North America WhatsApp pricing",
        "actual_state": (
            "Rates were found in the 'whatsapppricingrates' collection."
        ),
    })
    state = {
        "user_query": "What are the WhatsApp rates for North America?",
        "query_understanding": understanding,
        "evidence": [pricing, official_documentation],
        "investigation": result,
        "root_cause_analysis": RootCauseAnalysis(
            is_established=True,
            root_cause="A public pricing API is missing.",
            confidence=0.9,
            supporting_evidence_ids=["ev-pricing"],
            recommended_fix=["Build a new public API."],
        ),
    }
    assert route_after_investigation(state) == "build_result_report"
    report = build_final_report_node(state)["final_report"]
    assert report.response_type == "informational"
    assert report.root_cause is None
    assert report.recommended_fix == []
    assert "underlying cause" not in report.customer_response
    assert "whatsapppricingrates" not in report.customer_response
    assert report.customer_response.startswith(
        "Subject: TraceX information request update"
    )
    assert "Marketing: 2.1976" in report.customer_response
    assert "destination and billing context" in report.customer_response
    assert report.external_references[0].url == (
        "https://developers.facebook.com/docs/whatsapp/pricing"
    )
    assert "Reference documentation:" in report.customer_response


def test_organization_id_alone_does_not_verify_name_request():
    id_only = Evidence(
        id="ev-org-id",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="mongo-1",
        summary="grouped campaigns",
        content={
            "evidence_role": "final_answer",
            "error": None,
            "pipeline": [{"$sort": {"count": -1}}, {"$limit": 1}],
            "rows": [{"_id": "org-1", "count": 130}],
        },
    )
    state = {
        "user_query": "Which organization has run the highest number of campaigns?",
        "evidence": [id_only],
    }
    assert final_answer_evidence(state) is None

    named = id_only.model_copy(deep=True)
    named.content["rows"] = [{
        "organizationId": "org-1",
        "organizationName": "Example Organization",
        "count": 130,
    }]
    assert final_answer_evidence({
        **state,
        "evidence": [named],
    }) is not None


def test_verifier_is_domain_agnostic_for_new_entity_types():
    opaque = Evidence(
        id="ev-service",
        evidence_type=EvidenceType.DATABASE_QUERY,
        source="db-1",
        summary="service retries",
        content={
            "evidence_role": "final_answer",
            "error": None,
            "query": "SELECT service_id, retry_count ORDER BY retry_count DESC LIMIT 1",
            "rows": [{"serviceId": "svc-17", "retryCount": 42}],
        },
    )
    state = {
        "user_query": "Which service has the highest retry count?",
        "evidence": [opaque],
    }
    assert final_answer_evidence(state) is None
    named = opaque.model_copy(deep=True)
    named.content["rows"] = [{
        "serviceId": "svc-17",
        "serviceName": "notification-dispatcher",
        "retryCount": 42,
    }]
    assert final_answer_evidence({**state, "evidence": [named]}) is not None


def test_source_planner_starts_incident_with_runtime_sources_and_optional_code(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("LOG_ROOT", str(tmp_path))
    understanding = extract_business_entities_node({
        "user_query": "Why did bookingid BK-1042 fail with a timeout?"
    })["query_understanding"]
    update = plan_evidence_sources_node({
        "user_query": "Why did bookingid BK-1042 fail with a timeout?",
        "query_understanding": understanding,
    })
    assert set(update["evidence_source_plan"].sources) == {
        EvidenceSource.DATABASE, EvidenceSource.LOGS,
    }
    assert EvidenceSource.CODEBASE in update["evidence_source_plan"].optional_sources
    assert EvidenceSource.WEB in update["evidence_source_plan"].optional_sources


def test_source_planner_keeps_simple_retrieval_database_only():
    understanding = extract_business_entities_node({
        "user_query": "List users"
    })["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": "List users", "query_understanding": understanding
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.DATABASE]


def test_connected_repository_is_optional_for_database_retrieval():
    query = "List the latest ten accounts"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query,
        "query_understanding": understanding,
        "database_sources": [{"connection_id": "db-1"}],
        "codebase_sources": [{"connection_id": "repo-1"}],
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.DATABASE]
    assert plan.optional_sources == [EvidenceSource.CODEBASE]


def test_context_manifest_excludes_database_credentials():
    manifest = build_context_manifest_node({
        "investigation_id": "inv-1",
        "organization_id": "org-1",
        "project_id": "project-1",
        "database_sources": [{
            "connection_id": "db-1",
            "provider": "mongodb",
            "connection_url": "mongodb://secret@example.invalid/customer",
            "analysis": {"schema_hash": "schema-v1", "objects": []},
        }],
        "codebase_sources": [],
    })["context_manifest"]
    assert manifest["sources"][0]["version"] == "schema-v1"
    assert "secret" not in repr(manifest)


def test_database_skill_is_progressively_selected_for_database_plan():
    understanding = extract_business_entities_node({"user_query": "List users"})["query_understanding"]
    skills = select_skills(understanding, {EvidenceSource.DATABASE})
    assert [item.name for item in skills] == ["database-investigation"]


def test_mongodb_schema_type_coercion_does_not_depend_on_field_name():
    value = _coerce_mongodb_schema_types(
        {"objectType": "695653deffb2f9d2eccdd6d5"},
        {"objectType": "ObjectId"},
    )
    assert str(value["objectType"]) == "695653deffb2f9d2eccdd6d5"
    assert value["objectType"].__class__.__name__ == "ObjectId"


def test_mongodb_schema_type_coercion_applies_inside_operators():
    value = _coerce_mongodb_schema_types(
        {"owner": {"$in": ["695653deffb2f9d2eccdd6d5"]}},
        {"owner": "objectId"},
    )
    assert value["owner"]["$in"][0].__class__.__name__ == "ObjectId"


def test_mongodb_scalar_field_rejects_embedded_object_filter():
    with pytest.raises(ValueError, match="cannot be an embedded object"):
        _coerce_mongodb_schema_types(
            {"organisationId": {"_id": "695653deffb2f9d2eccdd6d1"}},
            {"organisationId": "ObjectId"},
        )


@pytest.mark.parametrize("wrapper", ["$oid", "'$oid'", '\"$oid\"'])
def test_mongodb_objectid_extended_json_wrappers_are_normalized(wrapper):
    value = _normalize_mongodb_extended_json({
        "organisationId": {wrapper: "695653deffb2f9d2eccdd6d1"}
    })
    assert value["organisationId"].__class__.__name__ == "ObjectId"
    assert str(value["organisationId"]) == "695653deffb2f9d2eccdd6d1"


def test_mongodb_extended_json_rejects_invalid_objectid():
    with pytest.raises(ValueError, match="valid ObjectId string"):
        _normalize_mongodb_extended_json({"ownerId": {"$oid": "invalid"}})


def test_duplicate_code_file_read_reuses_persisted_evidence(monkeypatch):
    calls = 0

    def read_file(path, ref, connection_id):
        nonlocal calls
        calls += 1
        return {"path": path, "content": "export class Campaign {}"}

    monkeypatch.setattr(evidence_tools, "_github_contents", read_file)

    async def exercise():
        token = bind_investigation("inv-deduplicate-code")
        try:
            first = await evidence_tools.get_codebase_file(
                "src/campaign/schemas/campaign.schema.ts"
            )
            second = await evidence_tools.get_codebase_file(
                "src/campaign/schemas/campaign.schema.ts"
            )
            saved = await evidence_repository.list_by_investigation(
                "inv-deduplicate-code"
            )
            assert first["result"] == second["result"]
            assert second["already_collected"] is True
            assert calls == 1
            assert len(saved) == 1
        finally:
            reset_investigation(token)
            await evidence_repository.clear("inv-deduplicate-code")

    asyncio.run(exercise())


def test_mongodb_tool_rejects_filter_alias_instead_of_scanning():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        evidence_agent.MongoQueryToolInput.model_validate({
            "collection": "organisations",
            "filter": {"name": "BIGBROS AI"},
            "purpose": "exploration",
        })


def test_unfiltered_find_cannot_be_marked_final_answer():
    with pytest.raises(ValueError, match="unfiltered collection sample"):
        TypedQueryIntent(
            object_name="wallets",
            operation="find",
            purpose="final_answer",
        )


def test_mongodb_ambiguous_string_filter_requires_value_discovery():
    with pytest.raises(ValueError, match="ambiguous analyzed types"):
        _coerce_mongodb_schema_types(
            {"reference": "42"},
            {"reference": "ObjectId | str"},
        )


def test_source_planner_supports_codebase_only_lookup():
    query = "Find the payment status enum in the code"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query, "query_understanding": understanding
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.CODEBASE]


def test_source_planner_combines_logs_and_database_for_concrete_id(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("LOG_ROOT", str(tmp_path))
    query = "Show logs for bookingid BK-9"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query, "query_understanding": understanding
    })["evidence_source_plan"]
    assert set(plan.sources) == {
        EvidenceSource.DATABASE, EvidenceSource.LOGS,
    }


def test_incident_without_connected_logs_never_plans_log_worker(monkeypatch):
    monkeypatch.delenv("LOG_ROOT", raising=False)
    query = "Payment failed"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query,
        "query_understanding": understanding,
        "database_sources": [{"connection_id": "db-1"}],
        "codebase_sources": [],
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.DATABASE]
    assert EvidenceSource.LOGS not in plan.optional_sources
    assert "not configured" in plan.unavailable_sources[EvidenceSource.LOGS]


def test_missing_log_root_is_not_reported_as_connected_capability(monkeypatch):
    monkeypatch.delenv("LOG_ROOT", raising=False)
    manifest = build_context_manifest_node({
        "investigation_id": "inv-no-logs",
        "organization_id": "org-1",
        "project_id": "project-1",
        "database_sources": [],
        "codebase_sources": [],
    })["context_manifest"]
    assert "logs" in manifest["unavailable_sources"]
    assert not any(item["source_type"] == "logs" for item in manifest["sources"])


def test_log_tool_cannot_be_created_without_log_capability(monkeypatch):
    monkeypatch.delenv("LOG_ROOT", raising=False)
    with pytest.raises(SourceUnavailableError, match="not configured"):
        evidence_agent.create_evidence_agent({EvidenceSource.LOGS})
