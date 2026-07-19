from __future__ import annotations

import os
import re
from time import monotonic
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

FORBIDDEN = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "MERGE",
    "COPY",
}


READ_ONLY_PREFIXES = ("SELECT", "SHOW", "EXPLAIN", "WITH")


def validate_read_only_query(query: str) -> None:
    normalized = re.sub(r"/\*.*?\*/|--[^\n]*", " ", query, flags=re.S).strip()
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed")
    upper = statements[0].upper()
    if not upper.startswith(READ_ONLY_PREFIXES):
        raise ValueError("Only read-only SQL statements are allowed")
    if any(re.search(rf"\b{word}\b", upper) for word in FORBIDDEN):
        raise ValueError("Query contains a forbidden SQL operation")


def apply_limit(query: str) -> str:
    if "LIMIT" in query.upper():
        return query

    query = query.rstrip(";")

    return f"{query} LIMIT 100;"


def run_safe_read_query(query: str) -> dict[str, Any]:
    """
    Execute a read-only PostgreSQL query.

    Rules:
    - Only SELECT statements.
    - Dangerous statements are rejected.
    - LIMIT 100 is automatically applied.
    """
    validate_read_only_query(query)
    query = apply_limit(query)

    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    conn = psycopg2.connect(database_url)

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = '15s'")
            started = monotonic()
            cursor.execute(query)

            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            return {
                "query": query,
                "columns": list(records[0]) if records else [],
                "rows": records,
                "row_count": len(records),
                "truncated": len(records) >= 100,
                "execution_time_ms": int((monotonic() - started) * 1000),
                "error": None,
            }

    finally:
        conn.close()
