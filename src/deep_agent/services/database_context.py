from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatabaseSource:
    connection_id: str
    provider: str
    connection_url: str
    analysis: dict[str, Any]


_database_sources: ContextVar[tuple[DatabaseSource, ...]] = ContextVar(
    "database_sources",
    default=(),
)
_control_plane_ids: ContextVar[frozenset[str]] = ContextVar(
    "control_plane_ids",
    default=frozenset(),
)


def bind_database_sources(
    sources: list[dict[str, Any]],
    control_plane_ids: set[str] | None = None,
) -> tuple[Token, Token]:
    parsed = tuple(
        DatabaseSource(
            connection_id=source["connection_id"],
            provider=source["provider"],
            connection_url=source["connection_url"],
            analysis=source["analysis"],
        )
        for source in sources
    )
    return (
        _database_sources.set(parsed),
        _control_plane_ids.set(frozenset(control_plane_ids or set())),
    )


def reset_database_sources(tokens: tuple[Token, Token]) -> None:
    source_token, identifiers_token = tokens
    _database_sources.reset(source_token)
    _control_plane_ids.reset(identifiers_token)


def database_sources() -> tuple[DatabaseSource, ...]:
    sources = _database_sources.get()
    if not sources:
        raise RuntimeError("No analyzed database is connected to this project")
    return sources


def database_source(connection_id: str | None = None) -> DatabaseSource:
    sources = database_sources()
    if connection_id:
        for source in sources:
            if source.connection_id == connection_id:
                return source
        raise ValueError(f"Unknown project database connection: {connection_id}")
    if len(sources) != 1:
        ids = ", ".join(source.connection_id for source in sources)
        raise ValueError(
            "Multiple databases are connected; provide connection_id. "
            f"Available connections: {ids}"
        )
    return sources[0]


def control_plane_ids() -> frozenset[str]:
    """IDs used to scope TraceX itself; they must never scope a client query."""
    return _control_plane_ids.get()
