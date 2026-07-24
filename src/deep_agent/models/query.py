from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    DATABASE = "database"
    CODEBASE = "codebase"
    LOGS = "logs"


class BusinessEntity(BaseModel):
    entity_type: str
    value: str | None = None
    source_text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class QueryUnderstanding(BaseModel):
    intent: str
    entities: list[BusinessEntity] = Field(default_factory=list)
    business_terms: list[str] = Field(default_factory=list)


class EvidenceSourcePlan(BaseModel):
    sources: list[EvidenceSource]
    reasons: dict[EvidenceSource, str] = Field(default_factory=dict)
