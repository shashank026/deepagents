from typing import Any, TypedDict

from deep_agent.models.evidence import Evidence
from deep_agent.models.investigation import InvestigationResult
from deep_agent.models.report import RootCauseReport
from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.models.query import EvidenceSourcePlan, QueryUnderstanding
from deep_agent.models.execution import (
    FailedAssumption,
    InvestigationPlanStep,
    ToolError,
)


class InvestigationState(TypedDict, total=False):
    investigation_id: str
    user_query: str
    organization_id: str
    project_id: str
    database_sources: list[dict[str, Any]]
    codebase_sources: list[dict[str, Any]]
    organization_knowledge: list[dict[str, Any]]
    extracted_entities: dict[str, Any]
    query_understanding: QueryUnderstanding | None
    evidence_source_plan: EvidenceSourcePlan | None
    investigation_plan: list[InvestigationPlanStep]
    repository_context: dict[str, Any]
    schema_context: dict[str, Any]
    evidence: list[Evidence]
    evidence_collection_attempts: int
    evidence_collection_errors: list[str]
    investigation: InvestigationResult | None
    root_cause_analysis: RootCauseAnalysis | None
    final_report: RootCauseReport | None
    requested_evidence: list[str]
    failed_assumptions: list[FailedAssumption]
    tool_errors: list[ToolError]
    retry_counts: dict[str, int]
    reasoning_calls: int
    insufficient_evidence: bool
    report_validation_errors: list[str]
    current_stage: str
    failure_reason: str | None
