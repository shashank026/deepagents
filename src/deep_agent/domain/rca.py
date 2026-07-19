from typing import Any, Optional

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default="", max_length=100)
    description: str = Field(default="", max_length=1_000)
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    contradicting_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    confidence: float = Field(default=0.0, ge=0, le=1)
    verification_required: list[str] = Field(
        default_factory=list,
        max_length=10,
    )


class InvestigationAnalysis(BaseModel):
    decision_summary: str = Field(default="", max_length=500)
    next_action_summary: str = Field(default="", max_length=120)
    continue_investigation: bool = False
    verification_required: list[str] = Field(
        default_factory=list,
        max_length=5,
    )
    expected_state: str = Field(default="", max_length=1_500)
    actual_state: str = Field(default="", max_length=1_500)
    timeline: list[str] = Field(default_factory=list, max_length=20)
    anomalies: list[str] = Field(default_factory=list, max_length=20)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    insufficient_evidence: bool = False
    missing_information: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

class ResultRecord(BaseModel):
    """
    A database record returned during the investigation.

    Store each database row in a controlled, serializable format.
    """

    source: str = Field(
        default="database",
        description="Source that produced this record.",
        max_length=100,
    )

    table_name: Optional[str] = Field(
        default=None,
        description="Database table or collection associated with the record.",
        max_length=200,
    )

    record_id: Optional[str] = Field(
        default=None,
        description="Primary identifier of the record when available.",
        max_length=500,
    )

    data: str = Field(
        default="",
        description=(
            "JSON-serialized representation of the record. "
            "Sensitive values must be masked."
        ),
        max_length=10_000,
    )


class RootCauseReport(BaseModel):
    issue_summary: str = Field(
        default="",
        description="Concise summary of the issue being investigated.",
        max_length=1_500,
    )

    root_cause: Optional[str] = Field(
        default=None,
        description=(
            "Most likely root cause supported by investigation evidence. "
            "Use null when the root cause cannot be established."
        ),
        max_length=2_000,
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence in the identified root cause, from 0 to 1.",
        ge=0.0,
        le=1.0,
    )

    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of evidence supporting the conclusion.",
        max_length=20,
    )

    contradicting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of evidence contradicting the conclusion.",
        max_length=20,
    )

    affected_components: list[str] = Field(
        default_factory=list,
        description="Applications, services, tables, APIs, or components affected.",
        max_length=20,
    )

    expected_state: str = Field(
        default="",
        description="Expected system or business state.",
        max_length=1_500,
    )

    actual_state: str = Field(
        default="",
        description="Observed system or business state.",
        max_length=1_500,
    )

    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Ordered actions recommended to resolve or investigate the issue.",
        max_length=20,
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Information still required to establish the root cause.",
        max_length=20,
    )

    customer_response: str = Field(
        default="",
        description="Clear, non-sensitive response suitable for the customer.",
        max_length=2_000,
    )

    engineering_note: str = Field(
        default="",
        description="Detailed technical note for the internal engineering team.",
        max_length=2_000,
    )

    result_records: list[ResultRecord] = Field(
        default_factory=list,
        description="Relevant records returned during the investigation.",
        max_length=100,
    )