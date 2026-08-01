from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceCapability(BaseModel):
    source_type: Literal["database", "codebase", "logs", "traces", "deployment", "web"]
    source_id: str
    provider: str | None = None
    environment: str | None = None
    version: str | None = None
    read_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvestigationLimits(BaseModel):
    max_evidence_rounds: int = Field(default=3, ge=1)
    max_rows_per_query: int = Field(default=100, ge=1)
    deadline_seconds: int | None = Field(default=None, ge=1)


class InvestigationContextManifest(BaseModel):
    """Immutable, secret-free description of one investigation's authority."""

    investigation_id: str
    organization_id: str
    project_id: str
    sources: list[SourceCapability] = Field(default_factory=list)
    permissions: dict[str, str] = Field(default_factory=lambda: {
        "database": "read_only",
        "codebase": "read_only",
        "runtime": "read_only",
        "shell": "unavailable",
    })
    limits: InvestigationLimits = Field(default_factory=InvestigationLimits)
    unavailable_sources: list[str] = Field(default_factory=list)
