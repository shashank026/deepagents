from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    DATABASE_SCHEMA = "database_schema"
    DATABASE_RECORD = "database_record"
    DATABASE_QUERY = "database_query"
    LOG_ENTRY = "log_entry"
    CODE_REFERENCE = "code_reference"
    API_RESPONSE = "api_response"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"


class EvidenceReliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Evidence(BaseModel):
    id: str
    evidence_type: EvidenceType
    source: str
    summary: str
    content: dict[str, Any] = Field(default_factory=dict)
    reliability: EvidenceReliability = EvidenceReliability.MEDIUM
    supports_hypothesis: list[str] = Field(default_factory=list)
    contradicts_hypothesis: list[str] = Field(default_factory=list)
    collected_at: str | None = None


class EvidenceCollectionResult(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    entities_found: dict[str, str] = Field(default_factory=dict)
    sources_searched: list[str] = Field(default_factory=list)
    queries_executed: list[str] = Field(default_factory=list)
    collection_summary: str = ""
    missing_sources: list[str] = Field(default_factory=list)
    collection_errors: list[str] = Field(default_factory=list)
