from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelCapabilities(BaseModel):
    supports_tool_calling: bool = True
    supports_native_structured_output: bool = True
    supports_json_schema: bool = True


class FailedAssumption(BaseModel):
    assumption: str
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    correction: str | None = None
    retryable: bool = True


class ToolError(BaseModel):
    tool_name: str
    error_code: str
    error_message: str
    retryable: bool = False
    input_summary: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    success: bool
    tool_name: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    data: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    execution_ms: int = Field(default=0, ge=0)


class InvestigationPlanStep(BaseModel):
    stage: Literal[
        "codebase", "schema", "database", "logs", "web", "validation"
    ]
    objective: str
    completion_criteria: str = ""
    status: Literal["pending", "completed", "blocked"] = "pending"
    evidence_ids: list[str] = Field(default_factory=list)
