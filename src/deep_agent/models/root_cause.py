from pydantic import BaseModel, Field


class RootCauseAnalysis(BaseModel):
    root_cause: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selected_hypothesis_id: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    is_established: bool = False
    missing_information: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    recommended_fix: list[str] = Field(default_factory=list)
    validation_steps: list[str] = Field(default_factory=list)
