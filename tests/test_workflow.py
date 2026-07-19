from deep_agent.models.evidence import Evidence, EvidenceType
from deep_agent.models.investigation import Hypothesis, InvestigationResult
from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.nodes.evidence_validation import route_after_evidence_validation, validate_evidence_node
from deep_agent.nodes.investigation import route_after_investigation
from deep_agent.nodes.report_builder import build_final_report_node
from deep_agent.nodes.root_cause_validation import validate_root_cause_node
from deep_agent.services.model_retry import is_rate_limit_error, retry_delay_seconds
from deep_agent.tools.database import apply_limit, validate_read_only_query
from deep_agent.tools.tools import retrieve_schema_context


def evidence():
    return Evidence(id="ev-1", evidence_type=EvidenceType.DATABASE_QUERY,
                    source="db", summary="one row",
                    content={"rows": [{"id": 1}], "row_count": 1})


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
        source="db", summary="statuses", content={"rows": [{"status": "S"}]},
    )
    final = Evidence(
        id="ev-bookings", evidence_type=EvidenceType.DATABASE_QUERY,
        source="db", summary="bookings",
        content={"rows": [{"bookingid": "1"}, {"bookingid": "2"}]},
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
    state = {
        "user_query": "Give me successful payments",
        "evidence": [evidence()],
        "investigation": investigation(),
    }
    assert route_after_investigation(state) == "build_result_report"


def test_sql_validation_rejects_writes_and_multiple_statements():
    validate_read_only_query("SELECT * FROM payments")
    assert apply_limit("SELECT 1") == "SELECT 1 LIMIT 100;"
    for query in ("DELETE FROM payments", "SELECT 1; DROP TABLE payments"):
        try:
            validate_read_only_query(query)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe SQL accepted")


def test_rate_limit_retry_delay_is_extracted():
    error = RuntimeError("429 RESOURCE_EXHAUSTED; Please retry in 52.063s")
    assert is_rate_limit_error(error)
    assert retry_delay_seconds(error) == 52.063


def test_vectorless_schema_retrieval_finds_domain_tables():
    matches = retrieve_schema_context("bookings with successful payments", limit=8)
    names = {item["name"] for item in matches}
    assert "payment" in names
    assert "orders" in names
