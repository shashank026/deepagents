from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.correction import SemanticResolution


class QueryFilter(BaseModel):
    field: str
    operator: Literal["eq", "in", "gt", "gte", "lt", "lte"]
    value: Any


class QuerySort(BaseModel):
    field: str
    direction: Literal["asc", "desc"]


class QueryIntent(BaseModel):
    object_name: str
    operation: Literal["find", "count", "aggregate", "distinct"]
    filters: list[QueryFilter] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    sort: list[QuerySort] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=500)
    is_exploratory: bool = False
    is_final_result: bool = False
    semantic_resolutions: list[SemanticResolution] = Field(default_factory=list)