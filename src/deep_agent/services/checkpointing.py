import os
import asyncio
from contextlib import asynccontextmanager


def _postgres_url(value: str) -> str:
    return value.replace("postgresql+asyncpg://", "postgresql://", 1).replace(
        "postgres+asyncpg://", "postgresql://", 1
    )


@asynccontextmanager
async def checkpoint_provider():
    """Yield a durable async checkpointer selected entirely by configuration."""
    url = os.getenv("CHECKPOINT_DATABASE_URL", "").strip()
    required = os.getenv("CHECKPOINT_REQUIRED", "false").lower() == "true"
    if not url:
        if required:
            raise RuntimeError(
                "CHECKPOINT_REQUIRED is true but CHECKPOINT_DATABASE_URL is empty"
            )
        yield None
        return
    if url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = url.split("///", 1)[-1]
        async with AsyncSqliteSaver.from_conn_string(path) as checkpointer:
            await _setup_once(url, checkpointer)
            yield checkpointer
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(
        _postgres_url(url)
    ) as checkpointer:
        await _setup_once(url, checkpointer)
        yield checkpointer
_setup_lock = asyncio.Lock()
_setup_urls: set[str] = set()


async def _setup_once(url: str, checkpointer) -> None:
    if url in _setup_urls:
        return
    async with _setup_lock:
        if url not in _setup_urls:
            await checkpointer.setup()
            _setup_urls.add(url)

