from typing import Any

from pydantic import BaseModel, Field


class ResultRecord(BaseModel):
    source: str
    record: dict[str, Any] = Field(default_factory=dict)


class RootCauseReport(BaseModel):
    response_type: str = "analysis"
    verification_status: str = "inconclusive"
    issue_summary: str = ""
    root_cause: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    expected_state: str = ""
    actual_state: str = ""
    suggested_actions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    customer_response: str = ""
    engineering_note: str = ""
    result_records: list[ResultRecord] = Field(default_factory=list)
