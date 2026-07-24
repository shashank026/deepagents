import asyncio
import pytest

from deep_agent.models.evidence import Evidence, EvidenceType
from deep_agent.models.investigation import Hypothesis, InvestigationResult
from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.nodes.evidence_validation import route_after_evidence_validation, validate_evidence_node
from deep_agent.nodes.investigation import route_after_investigation
from deep_agent.nodes.report_builder import build_final_report_node
from deep_agent.nodes.root_cause_validation import validate_root_cause_node
from deep_agent.nodes.query_understanding import extract_business_entities_node
from deep_agent.nodes.source_planning import plan_evidence_sources_node
from deep_agent.models.query import EvidenceSource
from deep_agent.services.model_retry import is_rate_limit_error, retry_delay_seconds
from deep_agent.services.database_context import bind_database_sources, reset_database_sources
from deep_agent.services.evidence_repository import (
    bind_investigation,
    evidence_repository,
    reset_investigation,
)
from deep_agent.services.retrieval_verification import final_answer_evidence
from deep_agent.tools import evidence_tools
from deep_agent.tools.database import (
    apply_limit,
    validate_read_only_query,
    _reject_control_plane_values,
    _normalize_sort_direction,
    _normalize_mongodb_sort,
    _bounded_mongodb_pipeline,
    _coerce_mongodb_ids,
    _validate_mongodb_value,
)
from deep_agent.tools import database as database_tools
from deep_agent.tools.tools import retrieve_schema_context


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


def test_evidence_validation_preserves_collection_error():
    update = validate_evidence_node({
        "evidence": [], "evidence_collection_errors": ["database unavailable"]
    })
    assert "database unavailable" in update["failure_reason"]


def test_root_cause_rejects_hallucinated_references():
    analysis = RootCauseAnalysis(root_cause="cause", confidence=.95, is_established=True,
        selected_hypothesis_id="made-up", supporting_evidence_ids=["made-up"])
    update = validate_root_cause_node({"evidence": [evidence()], "investigation": investigation(),
                                       "root_cause_analysis": analysis})
    assert update["root_cause_analysis"].is_established is False
    assert update["root_cause_analysis"].root_cause is None


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


def test_vectorless_schema_retrieval_finds_domain_tables():
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
        assert "orders" in names
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


def test_source_planner_selects_database_code_and_logs_for_incident():
    understanding = extract_business_entities_node({
        "user_query": "Why did bookingid BK-1042 fail with a timeout?"
    })["query_understanding"]
    update = plan_evidence_sources_node({
        "user_query": "Why did bookingid BK-1042 fail with a timeout?",
        "query_understanding": understanding,
    })
    assert set(update["evidence_source_plan"].sources) == {
        EvidenceSource.DATABASE, EvidenceSource.CODEBASE, EvidenceSource.LOGS
    }


def test_source_planner_keeps_simple_retrieval_database_only():
    understanding = extract_business_entities_node({
        "user_query": "List users"
    })["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": "List users", "query_understanding": understanding
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.DATABASE]


def test_source_planner_supports_codebase_only_lookup():
    query = "Find the payment status enum in the code"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query, "query_understanding": understanding
    })["evidence_source_plan"]
    assert plan.sources == [EvidenceSource.CODEBASE]


def test_source_planner_combines_logs_and_database_for_concrete_id():
    query = "Show logs for bookingid BK-9"
    understanding = extract_business_entities_node({"user_query": query})["query_understanding"]
    plan = plan_evidence_sources_node({
        "user_query": query, "query_understanding": understanding
    })["evidence_source_plan"]
    assert set(plan.sources) == {EvidenceSource.DATABASE, EvidenceSource.LOGS}
