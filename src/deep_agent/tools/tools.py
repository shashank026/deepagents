import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "database_analyzer.json"

with SCHEMA_PATH.open() as f:
    SCHEMA = json.load(f)


def list_tables():
    """Return all tables in the database."""
    return [
        f"{obj['namespace']}.{obj['name']}"
        for obj in SCHEMA["objects"]
    ]


def _compact_object(obj: dict) -> dict:
    """Return model-useful schema fields without analyzer bookkeeping noise."""
    return {
        "namespace": obj["namespace"],
        "name": obj["name"],
        "object_type": obj.get("object_type"),
        "description": obj.get("description"),
        "fields": [
            {
                "name": field.get("name"),
                "data_type": field.get("data_type"),
                "nullable": field.get("nullable"),
                "primary_key": field.get("is_primary_key", False),
                "foreign_key": field.get("is_foreign_key", False),
                "description": field.get("description"),
            }
            for field in obj.get("fields", [])
        ],
    }


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    synonyms = {
        "booking": {"order", "reservation"},
        "bookings": {"order", "reservation"},
        "payment": {"transaction"},
        "payments": {"transaction"},
        "successful": {"success", "status"},
        "failed": {"failure", "status"},
        "customer": {"client", "user"},
    }
    for token in tokens:
        expanded.update(synonyms.get(token, set()))
    return {token for token in expanded if len(token) > 1}


@lru_cache(maxsize=256)
def retrieve_schema_context(query: str, limit: int = 8) -> list[dict]:
    """Lexically rank relevant schemas without embeddings or model calls."""
    query_tokens = _tokens(query)
    ranked: list[tuple[int, str, dict]] = []
    for obj in SCHEMA["objects"]:
        name = obj["name"].lower()
        name_tokens = _tokens(name)
        field_tokens = {
            token
            for field in obj.get("fields", [])
            for token in _tokens(str(field.get("name", "")))
        }
        description_tokens = _tokens(str(obj.get("description") or ""))
        score = (
            12 * len(query_tokens & name_tokens)
            + 4 * len(query_tokens & field_tokens)
            + len(query_tokens & description_tokens)
        )
        if any(token in name for token in query_tokens):
            score += 6
        if score:
            ranked.append((score, f"{obj['namespace']}.{name}", obj))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    for _, _, obj in ranked:
        # Partition tables such as payment_2019_02 must not crowd every other
        # relevant schema out of a small retrieval window.
        family = re.sub(r"(?:_?20\d{2}(?:_?\d{2})*)$", "", obj["name"].lower())
        if family_counts.get(family, 0) >= 2:
            continue
        selected.append(_compact_object(obj))
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max(1, min(limit, 20)):
            break
    return selected


def get_table(table_name: str):
    """Return metadata for a specific table."""
    for obj in SCHEMA["objects"]:
        qualified_name = f"{obj['namespace']}.{obj['name']}"
        if obj["name"] == table_name or qualified_name == table_name:
            return _compact_object(obj)

    return {
        "error": f"Table '{table_name}' not found."
    }


def search_tables(keyword: str):
    """Search tables by keyword."""
    keyword = keyword.lower()

    results = []

    for obj in SCHEMA["objects"]:
        if keyword in obj["name"].lower():
            results.append(
                {
                    "namespace": obj["namespace"],
                    "name": obj["name"]
                }
            )

    return results
