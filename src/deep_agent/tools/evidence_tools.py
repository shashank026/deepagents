from datetime import datetime, timezone
from typing import Any
from typing import Literal
from uuid import uuid4

from deep_agent.models.evidence import Evidence, EvidenceReliability, EvidenceType
from deep_agent.services.evidence_repository import (
    active_investigation_id,
    evidence_repository,
)
from deep_agent.tools.database import run_safe_read_query as _run_query
from deep_agent.tools.database import run_safe_mongodb_query as _run_mongodb_query
from deep_agent.tools.tools import get_table as _get_table
from deep_agent.tools.tools import list_tables as _list_tables
from deep_agent.tools.tools import search_tables as _search_tables
from deep_agent.tools.tools import retrieve_schema_context as _retrieve_schema_context
from deep_agent.tools.external_sources import search_codebase_files, search_log_files


async def _save(source: str, kind: EvidenceType, summary: str, content: dict[str, Any]):
    evidence = Evidence(
        id=f"ev-{uuid4().hex}", evidence_type=kind, source=source,
        summary=summary, content=content, reliability=EvidenceReliability.HIGH,
        collected_at=datetime.now(timezone.utc).isoformat(),
    )
    await evidence_repository.save(active_investigation_id(), evidence)
    return content


async def list_database_objects() -> dict[str, Any]:
    """List database table names. Prefer retrieve_relevant_schema for user requests."""
    tables = _list_tables()
    return await _save("database_schema", EvidenceType.DATABASE_SCHEMA,
                       f"Found {len(tables)} database objects", {"objects": tables})


async def retrieve_relevant_schema(query: str, limit: int = 8) -> dict[str, Any]:
    """Retrieve locally ranked schemas relevant to a natural-language query.

    This uses lexical matching over table names, field names, and descriptions;
    it performs no model call and requires no vector database.
    """
    matches = _retrieve_schema_context(query, limit)
    return await _save(
        "project_database_analysis", EvidenceType.DATABASE_SCHEMA,
        f"Retrieved {len(matches)} schemas relevant to {query!r}",
        {"query": query, "matches": matches},
    )


async def get_table_schema(table_name: str) -> dict[str, Any]:
    """Get schema metadata for a table before constructing a query."""
    result = _get_table(table_name)
    return await _save(table_name, EvidenceType.DATABASE_SCHEMA,
                       f"Inspected schema for {table_name}", {"schema": result})


async def search_database_objects(keyword: str) -> dict[str, Any]:
    """Search database object names by keyword."""
    results = _search_tables(keyword)
    return await _save("database_schema", EvidenceType.DATABASE_SCHEMA,
                       f"Searched database objects for {keyword!r}", {"matches": results})


async def run_safe_read_query(
    query: str,
    connection_id: str | None = None,
    purpose: Literal["exploration", "final_answer"] = "exploration",
) -> dict[str, Any]:
    """Execute one validated read-only query, capped at 100 rows and 15 seconds."""
    try:
        result = _run_query(query, connection_id)
    except Exception as exc:
        await _save(
            connection_id or "project_database",
            EvidenceType.DATABASE_QUERY,
            f"Read-only query failed: {type(exc).__name__}",
            {
                "query": query,
                "evidence_role": purpose,
                "rows": [],
                "row_count": 0,
                "error": str(exc),
            },
        )
        raise
    result["evidence_role"] = purpose
    return await _save(
        connection_id or "project_database",
        EvidenceType.DATABASE_QUERY,
        f"Read-only query returned {result['row_count']} rows",
        result,
    )


async def run_safe_mongodb_query(
    collection: str,
    filter_query: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    sort: list[list[Any]] | dict[str, Any] | None = None,
    limit: int = 100,
    pipeline: list[dict[str, Any]] | None = None,
    connection_id: str | None = None,
    purpose: Literal["exploration", "final_answer"] = "exploration",
) -> dict[str, Any]:
    """Execute a read-only MongoDB find or aggregation, capped at 100 documents."""
    try:
        result = _run_mongodb_query(
            collection=collection,
            filter_query=filter_query,
            projection=projection,
            sort=sort,
            limit=limit,
            pipeline=pipeline,
            connection_id=connection_id,
        )
    except Exception as exc:
        await _save(
            connection_id or "project_database",
            EvidenceType.DATABASE_QUERY,
            f"MongoDB query failed: {type(exc).__name__}",
            {
                "collection": collection,
                "filter": filter_query,
                "projection": projection,
                "sort": sort,
                "pipeline": pipeline,
                "evidence_role": purpose,
                "rows": [],
                "row_count": 0,
                "error": str(exc),
            },
        )
        raise
    result["evidence_role"] = purpose
    return await _save(
        connection_id or "project_database",
        EvidenceType.DATABASE_QUERY,
        f"MongoDB query returned {result['row_count']} documents",
        result,
    )


async def search_codebase(query: str, max_results: int = 30) -> dict[str, Any]:
    """Search the configured codebase for mappings, enums, and application logic."""
    result = search_codebase_files(query, max_results)
    reliability = EvidenceType.CONFIGURATION if result.get("unavailable") else EvidenceType.CODE_REFERENCE
    return await _save(
        "codebase", reliability,
        result.get("error") or f"Code search returned {len(result['matches'])} matches",
        result,
    )


async def search_logs(query: str, max_results: int = 50) -> dict[str, Any]:
    """Search configured application logs for runtime events and failures."""
    result = search_log_files(query, max_results)
    return await _save(
        "logs", EvidenceType.LOG_ENTRY,
        result.get("error") or f"Log search returned {len(result['matches'])} matches",
        result,
    )
