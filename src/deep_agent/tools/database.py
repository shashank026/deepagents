from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from deep_agent.services.database_context import control_plane_ids, database_source


FORBIDDEN_SQL = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "MERGE", "COPY", "CALL", "EXEC", "EXECUTE",
}
READ_ONLY_PREFIXES = ("SELECT", "SHOW", "EXPLAIN", "WITH")
FORBIDDEN_MONGO_STAGES = {"$out", "$merge"}
FORBIDDEN_MONGO_OPERATORS = {"$where", "$function", "$accumulator"}
ALLOWED_MONGO_OPERATORS = {
    "$and", "$or", "$nor", "$not", "$eq", "$ne", "$gt", "$gte", "$lt",
    "$lte", "$in", "$nin", "$exists", "$type", "$regex", "$options",
    "$elemMatch", "$all", "$size", "$expr", "$literal", "$cond", "$ifNull",
    "$switch", "$case", "$then", "$else", "$toString", "$toObjectId",
    "$convert", "$dateFromString", "$match", "$project", "$group", "$sort",
    "$limit", "$skip", "$unwind", "$lookup", "$count", "$addFields", "$set",
    "$unset", "$replaceRoot", "$replaceWith", "$facet", "$bucket",
    "$bucketAuto", "$sortByCount", "$sample", "$sum", "$avg", "$min", "$max",
    "$first", "$last", "$push", "$addToSet", "$multiply", "$divide", "$subtract",
    "$add", "$concat", "$concatArrays", "$arrayElemAt", "$filter", "$map",
    "$oid", "$date", "$numberDecimal", "$numberLong",
}
MAX_ROWS = 100


def validate_read_only_query(query: str) -> None:
    normalized = re.sub(r"/\*.*?\*/|--[^\n]*", " ", query, flags=re.S).strip()
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed")
    upper = statements[0].upper()
    if not upper.startswith(READ_ONLY_PREFIXES):
        raise ValueError("Only read-only SQL statements are allowed")
    if any(re.search(rf"\b{word}\b", upper) for word in FORBIDDEN_SQL):
        raise ValueError("Query contains a forbidden SQL operation")


def apply_limit(query: str, provider: str = "postgresql") -> str:
    query = query.rstrip(";")
    if provider == "oracle":
        fetch = re.search(
            r"\bFETCH\s+FIRST\s+(\d+)\s+ROWS\s+ONLY\b",
            query,
            flags=re.IGNORECASE,
        )
        if fetch:
            requested = min(int(fetch.group(1)), MAX_ROWS)
            return (
                query[:fetch.start()]
                + f"FETCH FIRST {requested} ROWS ONLY"
                + query[fetch.end():]
            )
        return f"{query} FETCH FIRST {MAX_ROWS} ROWS ONLY"
    limit = re.search(r"\bLIMIT\s+(\d+)\b", query, flags=re.IGNORECASE)
    if limit:
        requested = min(int(limit.group(1)), MAX_ROWS)
        return query[:limit.start(1)] + str(requested) + query[limit.end(1):] + ";"
    if re.search(r"\bLIMIT\b", query, flags=re.IGNORECASE):
        raise ValueError("SQL LIMIT must be a numeric literal")
    return f"{query} LIMIT {MAX_ROWS};"


def run_safe_read_query(
    query: str,
    connection_id: str | None = None,
) -> dict[str, Any]:
    """Execute one read-only SQL query against PostgreSQL, MySQL, or Oracle."""
    source = database_source(connection_id)
    if source.provider == "mongodb":
        raise ValueError(
            "MongoDB does not accept SQL. Use run_safe_mongodb_query with a "
            "collection and filter or aggregation pipeline."
        )
    if source.provider not in {"postgresql", "mysql", "oracle"}:
        raise RuntimeError(f"Unsupported relational provider {source.provider!r}")

    validate_read_only_query(query)
    limited_query = apply_limit(query, source.provider)
    if source.provider == "postgresql":
        return _query_postgresql(source.connection_url, limited_query)
    if source.provider == "mysql":
        return _query_mysql(source.connection_url, limited_query)
    return _query_oracle(source.connection_url, limited_query)


def run_safe_mongodb_query(
    collection: str,
    filter_query: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    sort: list[list[Any]] | dict[str, Any] | None = None,
    limit: int = 100,
    pipeline: list[dict[str, Any]] | None = None,
    connection_id: str | None = None,
) -> dict[str, Any]:
    """Execute a bounded MongoDB find or aggregation operation."""
    source = database_source(connection_id)
    if source.provider != "mongodb":
        raise ValueError(
            f"run_safe_mongodb_query requires MongoDB, got {source.provider!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", collection):
        raise ValueError("Invalid MongoDB collection name")
    safe_limit = max(1, min(int(limit), MAX_ROWS))
    field_types = _mongodb_field_types(source, collection)
    # Check model-supplied strings before BSON normalization so protected
    # TraceX control-plane IDs cannot be hidden inside Extended JSON wrappers.
    _reject_control_plane_values(filter_query or {})
    _reject_control_plane_values(pipeline or [])
    normalized_filter = _normalize_mongodb_extended_json(filter_query or {})
    normalized_pipeline = _normalize_mongodb_extended_json(pipeline or [])
    typed_filter = _coerce_mongodb_schema_types(normalized_filter, field_types)
    typed_pipeline = _coerce_mongodb_pipeline_types(
        normalized_pipeline, field_types
    )
    _validate_mongodb_value(normalized_filter)
    _validate_mongodb_value(normalized_pipeline)
    _validate_mongodb_value(projection or {})
    _reject_control_plane_values(normalized_filter)
    _reject_control_plane_values(normalized_pipeline)

    from pymongo import MongoClient

    client = MongoClient(
        source.connection_url,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=15000,
    )
    started = monotonic()
    try:
        database = client.get_default_database()
        target = database[collection]
        if pipeline is not None:
            bounded_pipeline = _bounded_mongodb_pipeline(
                typed_pipeline,
                typed_filter,
                safe_limit,
            )
            rows = list(target.aggregate(bounded_pipeline, maxTimeMS=15000))
            operation: dict[str, Any] = {"pipeline": bounded_pipeline}
        else:
            cursor = target.find(
                typed_filter,
                projection,
                max_time_ms=15000,
            )
            if sort:
                normalized_sort = _normalize_mongodb_sort(sort)
                cursor = cursor.sort(normalized_sort)
            rows = list(cursor.limit(safe_limit))
            operation = {
                "filter": _json_safe(typed_filter),
                "projection": projection,
                "sort": normalized_sort if sort else None,
                "limit": safe_limit,
            }
        records = [_json_safe(row) for row in rows]
        return {
            "provider": "mongodb",
            "collection": collection,
            **operation,
            "columns": sorted({key for row in records for key in row}),
            "rows": records,
            "row_count": len(records),
            "truncated": len(records) >= safe_limit,
            "execution_time_ms": int((monotonic() - started) * 1000),
            "schema_validated": True,
            "field_types": field_types,
            "error": None,
        }
    finally:
        client.close()


def _query_postgresql(database_url: str, query: str) -> dict[str, Any]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    url = database_url.replace(
        "postgresql+asyncpg://", "postgresql://", 1
    ).replace("ssl=require", "sslmode=require")
    connection = psycopg2.connect(url)
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = '15s'")
            return _execute_cursor(cursor, query, "postgresql")
    finally:
        connection.close()


def _query_mysql(database_url: str, query: str) -> dict[str, Any]:
    import pymysql

    parsed = urlparse(database_url.replace("mysql+asyncmy://", "mysql://", 1))
    connection = pymysql.connect(
        host=parsed.hostname,
        port=parsed.port or 3306,
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=unquote(parsed.path.lstrip("/")),
        connect_timeout=5,
        read_timeout=15,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
        ssl={} if parse_qs(parsed.query).get("ssl", [""])[0] else None,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            return _execute_cursor(cursor, query, "mysql")
    finally:
        connection.rollback()
        connection.close()


def _query_oracle(database_url: str, query: str) -> dict[str, Any]:
    import oracledb

    parsed = urlparse(
        database_url.replace("oracle+oracledb_async://", "oracle://", 1)
    )
    params = parse_qs(parsed.query)
    service_name = params.get("service_name", [parsed.path.lstrip("/")])[0]
    dsn = oracledb.makedsn(
        parsed.hostname,
        parsed.port or 1521,
        service_name=service_name,
    )
    connection = oracledb.connect(
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        dsn=dsn,
    )
    connection.call_timeout = 15000
    try:
        with connection.cursor() as cursor:
            return _execute_cursor(cursor, query, "oracle")
    finally:
        connection.rollback()
        connection.close()


def _execute_cursor(cursor: Any, query: str, provider: str) -> dict[str, Any]:
    started = monotonic()
    cursor.execute(query)
    raw_rows = cursor.fetchall()
    if raw_rows and isinstance(raw_rows[0], dict):
        records = [_json_safe(dict(row)) for row in raw_rows]
        columns = list(records[0])
    else:
        columns = [str(item[0]).lower() for item in (cursor.description or [])]
        records = [
            {column: _json_safe(value) for column, value in zip(columns, row)}
            for row in raw_rows
        ]
    return {
        "provider": provider,
        "query": query,
        "columns": columns,
        "rows": records,
        "row_count": len(records),
        "truncated": len(records) >= MAX_ROWS,
        "execution_time_ms": int((monotonic() - started) * 1000),
        "error": None,
    }


def _validate_mongodb_value(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("MongoDB operation keys must be strings")
            stripped = key.strip("'\"")
            if "$" in key and (
                stripped != key or not key.startswith("$")
            ):
                raise ValueError(f"Malformed quoted MongoDB operator {key!r}")
            if key.startswith("$") and key not in ALLOWED_MONGO_OPERATORS:
                raise ValueError(f"Unsupported MongoDB operator {key!r}")
            if key in FORBIDDEN_MONGO_STAGES | FORBIDDEN_MONGO_OPERATORS:
                raise ValueError(f"MongoDB operation {key} is not allowed")
            _validate_mongodb_value(item)
    elif isinstance(value, list):
        for item in value:
            _validate_mongodb_value(item)


def _normalize_mongodb_extended_json(value: Any) -> Any:
    """Convert model-emitted Extended JSON ObjectId wrappers to BSON safely."""
    if isinstance(value, dict):
        if len(value) == 1:
            key, item = next(iter(value.items()))
            if isinstance(key, str) and key.strip("'\"") == "$oid":
                from bson import ObjectId

                if not isinstance(item, str) or not ObjectId.is_valid(item):
                    raise ValueError("MongoDB $oid requires a valid ObjectId string")
                return ObjectId(item)
        return {
            key: _normalize_mongodb_extended_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_mongodb_extended_json(item) for item in value]
    return value


def _normalize_sort_direction(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"desc", "descending", "-1"}:
            return -1
        if normalized in {"asc", "ascending", "1"}:
            return 1
    if value in {-1, 1}:
        return int(value)
    raise ValueError(
        "MongoDB sort direction must be asc/ascending/1 or desc/descending/-1"
    )


def _normalize_mongodb_sort(
    value: list[list[Any]] | dict[str, Any],
) -> list[tuple[str, int]]:
    items = list(value.items()) if isinstance(value, dict) else value
    normalized: list[tuple[str, int]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "MongoDB sort must be an object or a list of [field, direction] pairs"
            )
        field = str(item[0])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", field):
            raise ValueError(f"Invalid MongoDB sort field: {field!r}")
        normalized.append((field, _normalize_sort_direction(item[1])))
    if not normalized:
        raise ValueError("MongoDB sort cannot be empty")
    return normalized


def _bounded_mongodb_pipeline(
    pipeline: list[dict[str, Any]],
    filter_query: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    bounded = (
        [{"$match": _coerce_mongodb_ids(filter_query)}]
        if filter_query
        else []
    ) + [_coerce_mongodb_ids(stage) for stage in pipeline]
    for stage in bounded:
        if "$limit" in stage:
            try:
                stage["$limit"] = max(1, min(int(stage["$limit"]), limit))
            except (TypeError, ValueError) as exc:
                raise ValueError("MongoDB $limit must be an integer") from exc
    if not any("$limit" in stage for stage in bounded):
        bounded.append({"$limit": limit})
    return bounded


def _reject_control_plane_values(value: Any) -> None:
    protected = control_plane_ids()
    if not protected:
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_control_plane_values(item)
    elif isinstance(value, list):
        for item in value:
            _reject_control_plane_values(item)
    elif isinstance(value, str) and value in protected:
        raise ValueError(
            "A TraceX organization/project ID cannot be used as a client "
            "database filter. Discover the client-domain value from evidence."
        )


def _coerce_mongodb_ids(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        if key == "_id" and set(value) == {"_id"}:
            return _coerce_mongodb_ids(value["_id"], "_id")
        if set(value) == {"_oid"}:
            from bson import ObjectId
            return ObjectId(value["_oid"])
        if set(value) == {"$oid"}:
            from bson import ObjectId
            return ObjectId(value["$oid"])
        return {
            item_key: _coerce_mongodb_ids(item, item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_coerce_mongodb_ids(item, key) for item in value]
    normalized_key = key.lower().replace("_", "")
    if (
        isinstance(value, float)
        and normalized_key.endswith("id")
        and abs(value) >= 1_000_000_000_000_000
    ):
        raise ValueError(
            f"MongoDB identifier {key!r} must be passed as an exact string; "
            "floating-point/scientific notation would corrupt the value"
        )
    if (
        isinstance(value, str)
        and (key == "_id" or key.endswith("_id"))
        and re.fullmatch(r"[0-9a-fA-F]{24}", value)
    ):
        from bson import ObjectId
        return ObjectId(value)
    return value


def _mongodb_field_types(source: Any, collection: str) -> dict[str, str]:
    """Return analyzed BSON types for a collection without guessing a domain."""
    candidates = []
    for item in source.analysis.get("objects", []):
        name = str(item.get("name", ""))
        qualified = f"{item.get('namespace')}.{name}"
        if collection in {name, qualified}:
            candidates.append(item)
    if not candidates:
        from difflib import get_close_matches

        available = [
            str(item.get("name", ""))
            for item in source.analysis.get("objects", [])
            if item.get("name")
        ]
        suggestions = get_close_matches(collection, available, n=3, cutoff=0.5)
        hint = (
            f" Did you mean: {', '.join(repr(item) for item in suggestions)}?"
            if suggestions
            else ""
        )
        raise ValueError(
            f"Collection {collection!r} is absent from the analyzed schema."
            f"{hint} Use the exact name returned by schema discovery."
        )
    fields = candidates[0].get("fields", [])
    return {
        str(field.get("name")): str(field.get("data_type", "unknown"))
        for field in fields
        if field.get("name")
    }


def _coerce_scalar_for_mongodb_type(value: Any, data_type: str, field: str) -> Any:
    normalized = {
        part.strip().lower()
        for part in re.split(r"\||,", data_type)
        if part.strip()
    }
    # Mixed schemas are ambiguous. Keep an already native value, but do not
    # guess how a string should be represented when several types are observed.
    if len(normalized) > 1:
        if isinstance(value, str):
            raise ValueError(
                f"Field {field!r} has ambiguous analyzed types {data_type!r}; "
                "discover representative values before filtering it"
            )
        return value
    kind = next(iter(normalized), "unknown")
    if kind in {"objectid", "object_id"}:
        from bson import ObjectId
        if isinstance(value, ObjectId):
            return value
        if not isinstance(value, str) or not ObjectId.is_valid(value):
            raise ValueError(f"Field {field!r} requires a valid MongoDB ObjectId")
        return ObjectId(value)
    if kind in {"date", "datetime"} and isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Field {field!r} requires an ISO-8601 datetime") from exc
    if kind in {"int", "int32", "int64", "long"} and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field {field!r} requires an integer") from exc
    if kind in {"double", "float", "decimal", "decimal128"}:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field {field!r} requires a numeric value") from exc
    if kind in {"bool", "boolean"} and isinstance(value, str):
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"Field {field!r} requires true or false")
    return _coerce_mongodb_ids(value, field)


def _coerce_mongodb_schema_types(
    value: Any,
    field_types: dict[str, str],
    field: str | None = None,
) -> Any:
    if isinstance(value, dict):
        if field and field in field_types and value:
            keys = set(value)
            if not all(key.startswith("$") for key in keys):
                raise ValueError(
                    f"Field {field!r} is a scalar {field_types[field]!r}; "
                    "its filter value cannot be an embedded object"
                )
        converted = {}
        for key, item in value.items():
            next_field = field if key.startswith("$") else key
            converted[key] = _coerce_mongodb_schema_types(
                item, field_types, next_field
            )
        return converted
    if isinstance(value, list):
        return [
            _coerce_mongodb_schema_types(item, field_types, field)
            for item in value
        ]
    if field and field in field_types:
        return _coerce_scalar_for_mongodb_type(value, field_types[field], field)
    return _coerce_mongodb_ids(value, field or "")


def _coerce_mongodb_pipeline_types(
    pipeline: list[dict[str, Any]], field_types: dict[str, str]
) -> list[dict[str, Any]]:
    converted = []
    for stage in pipeline:
        if "$match" in stage:
            converted.append({
                **stage,
                "$match": _coerce_mongodb_schema_types(
                    stage["$match"], field_types
                ),
            })
        else:
            converted.append(_coerce_mongodb_ids(stage))
    return converted


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
