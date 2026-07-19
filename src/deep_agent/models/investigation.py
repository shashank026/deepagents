from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    id: str
    statement: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = Field(description="One of: proposed, supported, contradicted, inconclusive")
    validation_steps: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    issue_summary: str
    expected_state: str
    actual_state: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    requires_more_evidence: bool = False
    requested_evidence: list[str] = Field(default_factory=list)
