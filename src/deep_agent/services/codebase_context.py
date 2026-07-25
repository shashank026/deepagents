from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodebaseSource:
    connection_id: str
    provider: str
    installation_token: str
    api_url: str
    owner: str
    repository: str
    branch: str
    analysis: dict[str, Any]


_sources: ContextVar[tuple[CodebaseSource, ...]] = ContextVar(
    "codebase_sources", default=()
)


def bind_codebase_sources(sources: list[dict[str, Any]]) -> Token:
    return _sources.set(tuple(CodebaseSource(**source) for source in sources))


def reset_codebase_sources(token: Token) -> None:
    _sources.reset(token)


def codebase_source(connection_id: str | None = None) -> CodebaseSource:
    sources = _sources.get()
    if not sources:
        raise RuntimeError("No analyzed codebase is connected to this project")
    if connection_id:
        for source in sources:
            if source.connection_id == connection_id:
                return source
        raise ValueError(f"Unknown project codebase connection: {connection_id}")
    if len(sources) != 1:
        raise ValueError("Multiple codebases are connected; provide connection_id.")
    return sources[0]
