from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    DATABASE = "database"
    CODEBASE = "codebase"
    LOGS = "logs"
    WEB = "web"


class BusinessEntity(BaseModel):
    entity_type: str
    value: str | None = None
    source_text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class QueryConstraint(BaseModel):
    field_hint: str
    value: str
    source_text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class QueryUnderstanding(BaseModel):
    intent: str
    entities: list[BusinessEntity] = Field(default_factory=list)
    business_terms: list[str] = Field(default_factory=list)
    constraints: list[QueryConstraint] = Field(default_factory=list)
    requested_output: str = ""


class EvidenceSourcePlan(BaseModel):
    sources: list[EvidenceSource]
    reasons: dict[EvidenceSource, str] = Field(default_factory=dict)
    optional_sources: list[EvidenceSource] = Field(default_factory=list)
    escalation_reasons: dict[EvidenceSource, list[str]] = Field(
        default_factory=dict
    )
    unavailable_sources: dict[EvidenceSource, str] = Field(default_factory=dict)


class TypedQueryFilter(BaseModel):
    """Provider-neutral filter compiled against analyzed database metadata."""

    field: str
    operator: Literal[
        "eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "exists"
    ] = "eq"
    value: Any = None


class TypedQuerySort(BaseModel):
    field: str
    direction: Literal["asc", "desc"] = "asc"


class TypedQueryIntent(BaseModel):
    """A safe intermediate representation; it is not executable query text."""

    object_name: str
    operation: Literal["find", "count", "distinct"] = "find"
    filters: list[TypedQueryFilter] = Field(default_factory=list)
    projection: list[str] = Field(default_factory=list)
    sort: list[TypedQuerySort] = Field(default_factory=list)
    distinct_field: str | None = None
    limit: int = Field(default=100, ge=1, le=100)
    purpose: Literal[
        "exploration", "causal_validation", "final_answer"
    ] = "exploration"
