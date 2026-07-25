from langgraph.graph import END, START, StateGraph

from deep_agent.models.state import InvestigationState
from deep_agent.nodes.evidence_collection import collect_evidence_node
from deep_agent.nodes.evidence_validation import route_after_evidence_validation, validate_evidence_node
from deep_agent.nodes.investigation import investigate_node, route_after_investigation
from deep_agent.nodes.report_builder import build_final_report_node
from deep_agent.nodes.root_cause import identify_root_cause_node
from deep_agent.nodes.root_cause_validation import validate_root_cause_node
from deep_agent.nodes.query_understanding import extract_business_entities_node
from deep_agent.nodes.source_planning import plan_evidence_sources_node
from deep_agent.nodes.self_check import (
    revise_investigation_node,
    route_after_self_check,
    self_check_node,
)
from deep_agent.nodes.report_validation import validate_report_node


def create_investigation_graph():
    builder = StateGraph(InvestigationState)
    builder.add_node("extract_business_entities", extract_business_entities_node)
    builder.add_node("plan_evidence_sources", plan_evidence_sources_node)
    builder.add_node("collect_evidence", collect_evidence_node)
    builder.add_node("validate_evidence", validate_evidence_node)
    builder.add_node("investigate", investigate_node)
    builder.add_node("self_check", self_check_node)
    builder.add_node("revise_investigation", revise_investigation_node)
    builder.add_node("identify_root_cause", identify_root_cause_node)
    builder.add_node("validate_root_cause", validate_root_cause_node)
    builder.add_node("build_final_report", build_final_report_node)
    builder.add_node("validate_report", validate_report_node)
    builder.add_edge(START, "extract_business_entities")
    builder.add_edge("extract_business_entities", "plan_evidence_sources")
    builder.add_edge("plan_evidence_sources", "collect_evidence")
    builder.add_edge("collect_evidence", "validate_evidence")
    builder.add_conditional_edges("validate_evidence", route_after_evidence_validation, {
        "investigate": "investigate", "collect_more_evidence": "collect_evidence",
        "build_inconclusive_report": "build_final_report",
    })
    builder.add_edge("investigate", "self_check")
    builder.add_conditional_edges("self_check", route_after_self_check, {
        "revise_investigation": "revise_investigation",
        "identify_root_cause": "identify_root_cause",
        "build_inconclusive_report": "build_final_report",
        "build_result_report": "build_final_report",
    })
    builder.add_edge("revise_investigation", "collect_evidence")
    builder.add_edge("identify_root_cause", "validate_root_cause")
    builder.add_edge("validate_root_cause", "build_final_report")
    builder.add_edge("build_final_report", "validate_report")
    builder.add_edge("validate_report", END)
    return builder.compile()
