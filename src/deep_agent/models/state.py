from typing import Any, TypedDict

from deep_agent.models.evidence import Evidence
from deep_agent.models.investigation import InvestigationResult
from deep_agent.models.report import RootCauseReport
from deep_agent.models.root_cause import RootCauseAnalysis
from deep_agent.models.query import EvidenceSourcePlan, QueryUnderstanding


class InvestigationState(TypedDict, total=False):
    investigation_id: str
    user_query: str
    organization_id: str
    project_id: str
    database_sources: list[dict[str, Any]]
    extracted_entities: dict[str, Any]
    query_understanding: QueryUnderstanding | None
    evidence_source_plan: EvidenceSourcePlan | None
    evidence: list[Evidence]
    evidence_collection_attempts: int
    evidence_collection_errors: list[str]
    investigation: InvestigationResult | None
    root_cause_analysis: RootCauseAnalysis | None
    final_report: RootCauseReport | None
    requested_evidence: list[str]
    current_stage: str
    failure_reason: str | None
