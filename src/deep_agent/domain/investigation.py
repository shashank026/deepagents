from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float = Field(ge=0, le=1)


class IssueUnderstanding(BaseModel):
    request_type: Literal[
        "data_retrieval",
        "root_cause_investigation",
    ] = "root_cause_investigation"
    summary: str
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    affected_area: Optional[str] = None
    entities: list[ExtractedEntity] = Field(default_factory=list)
    required_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requested_limit: Optional[int] = Field(default=None, ge=1, le=500)
    requested_fields: list[str] = Field(default_factory=list, max_length=50)
    requested_conditions: list[str] = Field(default_factory=list, max_length=20)
    requested_ordering: list[str] = Field(default_factory=list, max_length=20)
    requested_grouping: list[str] = Field(default_factory=list, max_length=20)
    requested_count: Optional[int] = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)