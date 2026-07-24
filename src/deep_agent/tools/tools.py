import re
from deep_agent.services.database_context import database_sources


def _objects() -> list[dict]:
    objects: list[dict] = []
    for source in database_sources():
        relationships = source.analysis.get("relationships", [])
        for item in source.analysis.get("objects", []):
            objects.append({
                **item,
                "connection_id": source.connection_id,
                "provider": source.provider,
                "_relationships": [
                    relationship
                    for relationship in relationships
                    if (
                        relationship.get("source_object") == item.get("name")
                        or relationship.get("target_object") == item.get("name")
                    )
                ],
            })
    return objects


def list_tables():
    """Return all tables in the database."""
    return [
        f"{obj['namespace']}.{obj['name']}"
        for obj in _objects()
    ]


def _compact_object(obj: dict) -> dict:
    """Return model-useful schema fields without analyzer bookkeeping noise."""
    return {
        "namespace": obj["namespace"],
        "name": obj["name"],
        "object_type": obj.get("object_type"),
        "description": obj.get("description"),
        "relationships": obj.get("_relationships", []),
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


def retrieve_schema_context(query: str, limit: int = 8) -> list[dict]:
    """Lexically rank relevant schemas without embeddings or model calls."""
    query_tokens = _tokens(query)
    ranked: list[tuple[int, str, dict]] = []
    for obj in _objects():
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
        selected.append({
            **_compact_object(obj),
            "connection_id": obj["connection_id"],
            "provider": obj["provider"],
        })
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max(1, min(limit, 20)):
            break

    # Expand directly related objects while respecting the same small token
    # budget. This improves JOIN/$lookup construction without embedding every
    # schema in the model context.
    selected_keys = {
        (item["connection_id"], item["name"])
        for item in selected
    }
    object_lookup = {
        (obj["connection_id"], obj["name"]): obj
        for obj in _objects()
    }
    for item in list(selected):
        for relationship in item.get("relationships", []):
            related_name = (
                relationship.get("target_object")
                if relationship.get("source_object") == item["name"]
                else relationship.get("source_object")
            )
            key = (item["connection_id"], related_name)
            if not related_name or key in selected_keys or key not in object_lookup:
                continue
            related = object_lookup[key]
            selected.append({
                **_compact_object(related),
                "connection_id": related["connection_id"],
                "provider": related["provider"],
                "retrieval_reason": f"Related to {item['name']}",
            })
            selected_keys.add(key)
            if len(selected) >= max(1, min(limit, 20)):
                return selected
    return selected


def get_table(table_name: str):
    """Return metadata for a specific table."""
    for obj in _objects():
        qualified_name = f"{obj['namespace']}.{obj['name']}"
        if obj["name"] == table_name or qualified_name == table_name:
            return {
                **_compact_object(obj),
                "connection_id": obj["connection_id"],
                "provider": obj["provider"],
            }

    return {
        "error": f"Table '{table_name}' not found."
    }


def search_tables(keyword: str):
    """Search tables by keyword."""
    keyword = keyword.lower()

    results = []

    for obj in _objects():
        if keyword in obj["name"].lower():
            results.append(
                {
                    "namespace": obj["namespace"],
                    "name": obj["name"],
                    "connection_id": obj["connection_id"],
                    "provider": obj["provider"],
                }
            )

    return results
