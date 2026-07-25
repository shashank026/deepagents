import json
import os
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from deep_agent.models.execution import ModelCapabilities
from deep_agent.services.model_retry import invoke_with_rate_limit_retry
from deep_agent.stage_prompts import OUTPUT_REPAIR_PROMPT

T = TypeVar("T", bound=BaseModel)
STRUCTURED_OUTPUT_ERROR_MARKERS = (
    "response_format",
    "json_schema",
    "structured output",
    "unsupported",
)


def model_capabilities(model: Any) -> ModelCapabilities:
    mode = os.getenv("MODEL_STRUCTURED_OUTPUT_MODE", "auto").lower()
    if mode == "json":
        return ModelCapabilities(
            supports_tool_calling=True,
            supports_native_structured_output=False,
            supports_json_schema=True,
        )
    return ModelCapabilities()


async def invoke_structured(
    model: Any,
    schema: type[T],
    messages: list[dict[str, Any]],
    *,
    stage: str,
    max_repairs: int | None = None,
) -> T:
    capabilities = model_capabilities(model)
    if capabilities.supports_native_structured_output:
        try:
            runnable = model.with_structured_output(schema)
            return await invoke_with_rate_limit_retry(
                lambda: runnable.ainvoke(messages),
                stage=stage,
            )
        except Exception as exc:
            if not _is_capability_error(exc):
                raise

    repairs = (
        max_repairs
        if max_repairs is not None
        else int(os.getenv("MAX_OUTPUT_REPAIRS", "2"))
    )
    schema_text = json.dumps(schema.model_json_schema(), default=str)
    prompt_messages = [
        *messages,
        {
            "role": "system",
            "content": (
                "Native structured output is unavailable. Return JSON only. "
                f"Required JSON Schema:\n{schema_text}"
            ),
        },
    ]
    last_text = ""
    last_error = ""
    for attempt in range(repairs + 1):
        response = await invoke_with_rate_limit_retry(
            lambda: model.ainvoke(prompt_messages),
            stage=stage,
        )
        last_text = _message_text(response)
        try:
            return schema.model_validate(_extract_json(last_text))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt >= repairs:
                break
            prompt_messages.extend([
                {"role": "assistant", "content": last_text},
                {
                    "role": "user",
                    "content": (
                        f"{OUTPUT_REPAIR_PROMPT}\n\nValidation error:\n"
                        f"{last_error}\n\nJSON Schema:\n{schema_text}"
                    ),
                },
            ])
    raise ValueError(
        f"{stage} output failed Pydantic validation after "
        f"{repairs + 1} attempts: {last_error}"
    )


def _is_capability_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in STRUCTURED_OUTPUT_ERROR_MARKERS)


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


def _extract_json(value: str) -> Any:
    stripped = value.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start:end + 1])
