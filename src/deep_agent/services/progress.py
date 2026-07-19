from collections.abc import Callable
from contextvars import ContextVar, Token


_progress_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "progress_sink", default=None
)


def bind_progress_sink(sink: Callable[[str], None]) -> Token:
    return _progress_sink.set(sink)


def reset_progress_sink(token: Token) -> None:
    _progress_sink.reset(token)


def notify_progress(message: str) -> None:
    sink = _progress_sink.get()
    if sink is not None:
        sink(message)
