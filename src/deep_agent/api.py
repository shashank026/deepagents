from datetime import datetime, timezone
import json
from asyncio import to_thread
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, SecretStr

from deep_agent.main import NEXT_STAGE_MESSAGES, STAGE_MESSAGES, stream_investigation


class DatabaseSourceRequest(BaseModel):
    connection_id: str
    provider: str
    connection_url: SecretStr
    analysis: dict[str, Any]


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    organization_id: str
    project_id: str
    database_sources: list[DatabaseSourceRequest] = Field(min_length=1)
    callback_url: str | None = None
    callback_token: SecretStr | None = None


app = FastAPI(title="TraceX DeepAgents")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _send_progress(url: str, token: str, payload: dict[str, Any]) -> None:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    with urlopen(request, timeout=5):
        pass


def _safe(value: Any, depth: int = 0) -> Any:
    """Create a bounded, secret-free representation for persisted diagnostics."""
    if depth > 8:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if any(term in str(key).lower() for term in ("url", "password", "secret", "token"))
                else _safe(item, depth + 1)
            )
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in list(value)[:30]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:4000] if isinstance(value, str) else value
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump(mode="json"), depth + 1)
    return str(value)[:4000]


class InvestigationTraceCallback(BaseCallbackHandler):
    def __init__(self, sources: list[DatabaseSourceRequest]) -> None:
        self.events: list[dict[str, Any]] = []
        self.sources = {source.connection_id: source.provider for source in sources}
        self.default_connection_id = sources[0].connection_id if len(sources) == 1 else None

    def _append(self, event_type: str, title: str, **details: Any) -> None:
        self.events.append({
            "id": f"event-{len(self.events) + 1}",
            "timestamp": _now(),
            "type": event_type,
            "title": title,
            "details": _safe(details),
        })

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        model_id = serialized.get("id") or ["model"]
        model_name = model_id[-1] if isinstance(model_id, list) else str(model_id)
        self._append(
            "llm",
            "LLM request started",
            model=serialized.get("name") or model_name,
            message_count=sum(len(batch) for batch in messages),
        )

    def on_llm_end(self, response, **kwargs) -> None:
        self._append(
            "llm",
            "LLM request completed",
            usage=getattr(response, "llm_output", None) or {},
            generations=len(getattr(response, "generations", []) or []),
        )

    def on_llm_error(self, error, **kwargs) -> None:
        self._append("error", "LLM request failed", error=type(error).__name__)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name") or "tool"
        tool_input = kwargs.get("inputs") or input_str
        if isinstance(tool_input, str):
            try:
                parsed_input = json.loads(tool_input)
                if isinstance(parsed_input, dict):
                    tool_input = parsed_input
            except (TypeError, json.JSONDecodeError):
                pass
        connection_id = None
        if isinstance(tool_input, dict):
            connection_id = tool_input.get("connection_id")
        connection_id = connection_id or self.default_connection_id
        self._append(
            "database" if name in {
                "get_table_schema",
                "search_database_objects",
                "retrieve_relevant_schema",
                "run_safe_read_query",
                "run_safe_mongodb_query",
            } else "tool",
            f"Tool started: {name}",
            tool=name,
            input=tool_input,
            connection_id=connection_id,
            provider=self.sources.get(connection_id),
        )

    def on_tool_end(self, output, **kwargs) -> None:
        self._append("tool", "Tool completed", output=output)

    def on_tool_error(self, error, **kwargs) -> None:
        self._append("error", "Tool failed", error=type(error).__name__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/investigations")
async def investigate(payload: InvestigationRequest) -> dict[str, Any]:
    try:
        sources = [
                {
                    **source.model_dump(exclude={"connection_url"}),
                    "connection_url": source.connection_url.get_secret_value(),
                }
                for source in payload.database_sources
        ]
        callback = InvestigationTraceCallback(payload.database_sources)
        report = None
        evidence: list[Any] = []
        timeline: list[dict[str, Any]] = []
        stage_started_at = _now()
        async for event in stream_investigation(
            payload.question,
            payload.organization_id,
            payload.project_id,
            database_sources=sources,
            callbacks=[callback],
        ):
            for node_name, update in event.items():
                completed_at = _now()
                completed_label = STAGE_MESSAGES.get(
                    node_name,
                    node_name.replace("_", " ").title(),
                )
                timeline.append({
                    "id": f"stage-{len(timeline) + 1}",
                    "timestamp": completed_at,
                    "started_at": stage_started_at,
                    "completed_at": completed_at,
                    "stage": node_name,
                    "label": completed_label,
                    "status": "completed",
                })
                current_stage = update.get("current_stage") or ""
                next_label = NEXT_STAGE_MESSAGES.get(
                    current_stage,
                    str(current_stage).replace("_", " ").title(),
                )
                if payload.callback_url and payload.callback_token:
                    try:
                        await to_thread(
                            _send_progress,
                            payload.callback_url,
                            payload.callback_token.get_secret_value(),
                            {
                                "completed_stage": node_name,
                                "completed_label": completed_label,
                                "next_stage": current_stage,
                                "next_label": next_label,
                                "started_at": stage_started_at,
                                "completed_at": completed_at,
                                "timestamp": completed_at,
                            },
                        )
                    except Exception:
                        # Diagnostics must not make the investigation itself fail.
                        pass
                stage_started_at = completed_at
                if update.get("final_report") is not None:
                    report = update["final_report"]
                if update.get("evidence") is not None:
                    evidence = update["evidence"]
        if report is None:
            raise RuntimeError("Workflow completed without a final report")
        return {
            **report.model_dump(mode="json"),
            "_timeline": timeline,
            "_events": callback.events,
            "_evidence": [_safe(item) for item in evidence],
            "_databases": [
                {
                    "connection_id": source.connection_id,
                    "provider": source.provider,
                    "analysis_object_count": len(source.analysis.get("objects", [])),
                }
                for source in payload.database_sources
            ],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Investigation failed: {type(exc).__name__}",
        ) from exc
