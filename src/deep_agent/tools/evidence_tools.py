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
from deep_agent.tools.github import (
    get_blob as _github_blob, get_commit as _github_commit,
    get_contents as _github_contents, get_tree as _github_tree,
    inspect_symbol as _github_inspect_symbol, search_code as _github_search,
)
from deep_agent.services.database_context import database_source
from deep_agent.models.query import TypedQueryIntent
from deep_agent.tools.web_research import (
    fetch_public_page as _fetch_public_page,
    search_public_web as _search_public_web,
)


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


async def discover_field_values(
    object_name: str,
    field_name: str,
    connection_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Discover representative stored values for one schema-verified field."""
    schema = _get_table(object_name)
    if schema.get("error"):
        raise ValueError(schema["error"])
    known_fields = {
        field.get("name")
        for field in schema.get("fields", [])
    }
    if field_name not in known_fields:
        raise ValueError(
            f"Field {field_name!r} is not present in analyzed schema for "
            f"{object_name!r}"
        )
    source_id = connection_id or schema.get("connection_id")
    source = database_source(source_id)
    bounded_limit = max(1, min(limit, 50))
    if source.provider == "mongodb":
        result = _run_mongodb_query(
            collection=schema["name"],
            pipeline=[
                {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": bounded_limit},
                {"$project": {
                    "_id": 0,
                    "value": "$_id",
                    "count": 1,
                }},
            ],
            connection_id=source.connection_id,
        )
    else:
        if not all(
            part.replace("_", "").isalnum()
            for part in object_name.split(".")
        ) or not field_name.replace("_", "").isalnum():
            raise ValueError("Unsafe database identifier")
        quote = "`" if source.provider == "mysql" else '"'
        qualified = ".".join(
            f"{quote}{part}{quote}" for part in object_name.split(".")
        )
        column = f"{quote}{field_name}{quote}"
        result = _run_query(
            (
                f"SELECT {column} AS value, COUNT(*) AS count "
                f"FROM {qualified} GROUP BY {column} "
                f"ORDER BY count DESC"
            ),
            source.connection_id,
        )
    return await _save(
        source.connection_id,
        EvidenceType.DATABASE_RECORD,
        f"Discovered representative values for {object_name}.{field_name}",
        {
            **result,
            "object_name": object_name,
            "field_name": field_name,
            "semantic_discovery": True,
        },
    )


def _typed_mongodb_operation(intent: TypedQueryIntent) -> dict[str, Any]:
    operators = {
        "ne": "$ne", "in": "$in", "nin": "$nin", "gt": "$gt",
        "gte": "$gte", "lt": "$lt", "lte": "$lte", "exists": "$exists",
    }
    filters: dict[str, Any] = {}
    for item in intent.filters:
        if item.operator == "eq":
            filters[item.field] = item.value
        else:
            filters.setdefault(item.field, {})[operators[item.operator]] = item.value
    projection = {field: 1 for field in intent.projection} or None
    sort = [[item.field, item.direction] for item in intent.sort] or None
    if intent.operation == "count":
        return {"pipeline": [{"$match": filters}, {"$count": "count"}]}
    if intent.operation == "distinct":
        if not intent.distinct_field:
            raise ValueError("distinct operation requires distinct_field")
        return {"pipeline": [
            {"$match": filters},
            {"$group": {"_id": f"${intent.distinct_field}"}},
            {"$project": {"_id": 0, "value": "$_id"}},
        ]}
    return {"filter_query": filters, "projection": projection, "sort": sort}


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _typed_sql(intent: TypedQueryIntent, provider: str, fields: set[str]) -> str:
    quote = "`" if provider == "mysql" else '"'
    identifier = lambda value: quote + value.replace(quote, quote * 2) + quote
    selected = intent.projection or sorted(fields)
    if intent.operation == "find" and not selected:
        raise ValueError("Analyzed schema has no selectable fields")
    if intent.operation == "count":
        select = "COUNT(*) AS count"
    elif intent.operation == "distinct":
        if not intent.distinct_field:
            raise ValueError("distinct operation requires distinct_field")
        select = f"DISTINCT {identifier(intent.distinct_field)} AS value"
    else:
        select = ", ".join(identifier(field) for field in selected)
    operator = {
        "eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
    }
    clauses = []
    for item in intent.filters:
        column = identifier(item.field)
        if item.operator in {"in", "nin"}:
            if not isinstance(item.value, list) or not item.value:
                raise ValueError(f"{item.operator} requires a non-empty list")
            keyword = "IN" if item.operator == "in" else "NOT IN"
            clauses.append(
                f"{column} {keyword} ({', '.join(_sql_literal(v) for v in item.value)})"
            )
        elif item.operator == "exists":
            clauses.append(f"{column} IS {'NOT ' if item.value else ''}NULL")
        elif item.value is None and item.operator in {"eq", "ne"}:
            clauses.append(f"{column} IS {'NOT ' if item.operator == 'ne' else ''}NULL")
        else:
            clauses.append(f"{column} {operator[item.operator]} {_sql_literal(item.value)}")
    query = f"SELECT {select} FROM {'.'.join(identifier(p) for p in intent.object_name.split('.'))}"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    if intent.sort and intent.operation == "find":
        query += " ORDER BY " + ", ".join(
            f"{identifier(item.field)} {item.direction.upper()}" for item in intent.sort
        )
    return query


async def execute_typed_database_query(intent: TypedQueryIntent) -> dict[str, Any]:
    """Compile a provider-neutral intent against analyzed schema, then execute it."""
    schema = _get_table(intent.object_name)
    if schema.get("error"):
        raise ValueError(schema["error"])
    fields = {str(item.get("name")) for item in schema.get("fields", [])}
    referenced = {
        *intent.projection,
        *(item.field for item in intent.filters),
        *(item.field for item in intent.sort),
        *([intent.distinct_field] if intent.distinct_field else []),
    }
    unknown = sorted(field for field in referenced if field not in fields)
    if unknown:
        raise ValueError(f"Fields are absent from analyzed schema: {', '.join(unknown)}")
    source = database_source(schema.get("connection_id"))
    if source.provider == "mongodb":
        operation = _typed_mongodb_operation(intent)
        return await run_safe_mongodb_query(
            collection=schema["name"],
            limit=intent.limit,
            connection_id=source.connection_id,
            purpose=intent.purpose,
            **operation,
        )
    query = _typed_sql(intent, source.provider, fields)
    return await run_safe_read_query(
        query=query,
        connection_id=source.connection_id,
        purpose=intent.purpose,
    )


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
    """Search the connected GitHub repository for application logic."""
    try:
        result = _github_search(query, max_results=max_results)
    except RuntimeError as exc:
        if "No analyzed codebase" not in str(exc):
            raise
        result = search_codebase_files(query, max_results)
    reliability = EvidenceType.CONFIGURATION if result.get("unavailable") else EvidenceType.CODE_REFERENCE
    return await _save(
        "codebase", reliability,
        result.get("error") or f"Code search returned {len(result['matches'])} matches",
        result,
    )


async def get_codebase_file(path: str, ref: str | None = None,
                            connection_id: str | None = None) -> dict[str, Any]:
    """Read a GitHub file or directory by path using the Contents API."""
    result = _github_contents(path, ref, connection_id)
    return await _save(
        connection_id or "github", EvidenceType.CODE_REFERENCE,
        f"Read codebase path {path}", {"result": result},
    )


async def get_codebase_commit(ref: str | None = None,
                              connection_id: str | None = None) -> dict[str, Any]:
    """Read immutable commit metadata for a branch, tag, or commit SHA."""
    result = _github_commit(ref, connection_id)
    return await _save(
        connection_id or "github", EvidenceType.CODE_REFERENCE,
        f"Read commit {ref or 'configured branch'}", {"commit": result},
    )


async def get_codebase_tree(sha: str, recursive: bool = True,
                            connection_id: str | None = None) -> dict[str, Any]:
    """List repository files and paths from a Git tree SHA."""
    result = _github_tree(sha, recursive, connection_id)
    return await _save(
        connection_id or "github", EvidenceType.CODE_REFERENCE,
        f"Read Git tree {sha}", result,
    )


async def get_codebase_blob(sha: str,
                            connection_id: str | None = None) -> dict[str, Any]:
    """Read one repository file by its immutable Git blob SHA."""
    result = _github_blob(sha, connection_id)
    return await _save(
        connection_id or "github", EvidenceType.CODE_REFERENCE,
        f"Read Git blob {sha}", result,
    )


async def inspect_codebase_symbol(
    symbol: str,
    path: str | None = None,
    connection_id: str | None = None,
    context_lines: int = 12,
) -> dict[str, Any]:
    """Persist focused source snippets for a call, constant, model, or field.

    Supply ``path`` to inspect a known caller. Omit it to use GitHub search to
    locate definitions and references. Use this to follow a service call into
    its implementation instead of stopping at the calling file.
    """
    result = _github_inspect_symbol(
        symbol,
        path=path,
        connection_id=connection_id,
        context_lines=context_lines,
    )
    return await _save(
        connection_id or "github",
        EvidenceType.CODE_REFERENCE,
        (
            f"Inspected symbol {symbol!r}; "
            f"found {result['match_count']} focused snippets"
        ),
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


async def search_public_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search allowlisted public technical sources for supporting context.

    Never include customer identifiers, source code, credentials, internal
    hostnames, or connection details. External results cannot independently
    establish a customer-specific root cause.
    """
    result = _search_public_web(query, max_results)
    return await _save(
        "public_web",
        EvidenceType.API_RESPONSE,
        (
            result.get("error")
            or f"Public web search returned {len(result.get('citations', []))} citations"
        ),
        {**result, "external_context_only": True},
    )


async def fetch_public_page(url: str) -> dict[str, Any]:
    """Fetch an allowlisted public documentation page as supporting context."""
    result = _fetch_public_page(url)
    return await _save(
        result.get("url", "public_web"),
        EvidenceType.API_RESPONSE,
        result.get("error") or f"Fetched public documentation page {url}",
        {**result, "external_context_only": True},
    )
