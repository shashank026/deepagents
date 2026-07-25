"""Shared, bounded reasoning-model access for analytical workflow stages."""

import asyncio
import logging
import os
from functools import lru_cache
from time import monotonic
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from deep_agent.services.structured_output import invoke_structured

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ReasoningService:
    """Own one model client and enforce process-wide reasoning limits."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        max_concurrency: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = os.getenv(
            "REASONING_MODEL",
            "gemini-3.1-flash-lite",
        )
        self.model = model or ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
        )
        concurrency = max_concurrency or int(
            os.getenv("REASONING_MAX_CONCURRENCY", "8")
        )
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("REASONING_TIMEOUT_SECONDS", "120")
        )

    async def invoke(
        self,
        schema: type[T],
        messages: list[dict[str, Any]],
        *,
        stage: str,
    ) -> T:
        started = monotonic()
        async with self._semaphore:
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    return await invoke_structured(
                        self.model,
                        schema,
                        messages,
                        stage=stage,
                    )
            finally:
                logger.info(
                    "reasoning_stage_completed stage=%s model=%s elapsed_ms=%d",
                    stage,
                    self.model_name,
                    int((monotonic() - started) * 1000),
                )


@lru_cache(maxsize=1)
def reasoning_service() -> ReasoningService:
    return ReasoningService()


def reasoning_call_limit() -> int:
    return max(
        1,
        int(os.getenv("REASONING_MAX_CALLS_PER_INVESTIGATION", "6")),
    )
