import asyncio
import os
import random
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from deep_agent.services.progress import notify_progress

T = TypeVar("T")


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def retry_delay_seconds(exc: BaseException, default: float = 60.0) -> float:
    text = str(exc)
    patterns = (
        r"retry in\s+(\d+(?:\.\d+)?)s?",
        r"retryDelay['\"\s:]+(\d+(?:\.\d+)?)s",
        r"retry after\s+(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return max(1.0, float(match.group(1)))
    return default


async def invoke_with_rate_limit_retry(
    operation: Callable[[], Awaitable[T]], *, stage: str,
) -> T:
    retries = max(0, int(os.getenv("MODEL_RATE_LIMIT_RETRIES", "2")))
    for attempt in range(retries + 1):
        try:
            return await operation()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= retries:
                raise
            delay = retry_delay_seconds(exc) + random.uniform(0.5, 1.5)
            notify_progress(
                f"{stage} hit the model quota; retrying in {delay:.0f}s "
                f"({attempt + 1}/{retries})"
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")
