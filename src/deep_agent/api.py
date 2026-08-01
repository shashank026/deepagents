from datetime import datetime, timezone
import asyncio
import json
import logging
import os
import secrets
from asyncio import Task, to_thread
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, Header, HTTPException
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, SecretStr

from deep_agent.main import NEXT_STAGE_MESSAGES, STAGE_MESSAGES, stream_investigation
from deep_agent.services.evidence_repository import evidence_repository


class DatabaseSourceRequest(BaseModel):
    connection_id: str
    provider: str
    connection_url: SecretStr
    analysis: dict[str, Any]


class CodebaseSourceRequest(BaseModel):
    connection_id: str
    provider: str
    installation_token: SecretStr
    api_url: str = "https://api.github.com"
    owner: str
    repository: str
    branch: str
    analysis: dict[str, Any]


class InvestigationRequest(BaseModel):
    execution_id: str | None = None
    execution_attempt: int = Field(default=1, ge=1)
    question: str = Field(min_length=3, max_length=4000)
    organization_id: str
    project_id: str
    database_sources: list[DatabaseSourceRequest] = Field(default_factory=list)
    codebase_sources: list[CodebaseSourceRequest] = Field(default_factory=list)
    organization_knowledge: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=10,
    )
    callback_url: str | None = None
    completion_url: str | None = None
    callback_token: SecretStr | None = None
    max_runtime_seconds: int = Field(default=840, ge=60, le=3600)


app = FastAPI(title="TraceX DeepAgents")
logger = logging.getLogger("uvicorn.error")
_active_investigations: dict[str, tuple[Task[Any], str]] = {}


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


def _send_completion(
    url: str, token: str, result: dict[str, Any], execution_attempt: int
) -> None:
    request = Request(
        url,
        data=json.dumps({
            "result": result,
            "execution_attempt": execution_attempt,
        }).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    with urlopen(request, timeout=15):
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


def _final_report_payload(report: Any, question: str) -> dict[str, Any]:
    if report is None:
        return {
            "investigation_status": "insufficient_evidence",
            "response_type": "incident",
            "verification_status": "inconclusive",
            "issue_summary": question,
            "root_cause": None,
            "confidence": 0.0,
            "customer_response": (
                "The investigation completed, but its final structured report "
                "could not be packaged. The collected evidence and workflow "
                "trace are preserved for review."
            ),
            "engineering_note": (
                "Workflow reached END without exposing final_report in the "
                "stream update."
            ),
            "missing_information": [
                "Final report stream payload was unavailable."
            ],
        }
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    if isinstance(report, dict):
        return _safe(report)
    raise TypeError(f"Unsupported final report type: {type(report).__name__}")


class InvestigationTraceCallback(BaseCallbackHandler):
    def __init__(self, sources: list[DatabaseSourceRequest],
                 codebases: list[CodebaseSourceRequest] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.sources = {source.connection_id: source.provider for source in sources}
        self.sources.update({
            source.connection_id: source.provider for source in (codebases or [])
        })
        all_sources = [*sources, *(codebases or [])]
        self.default_connection_id = (
            all_sources[0].connection_id if len(all_sources) == 1 else None
        )

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
                "discover_field_values",
                "run_safe_read_query",
                "run_safe_mongodb_query",
            } else "codebase" if name in {
                "search_codebase", "get_codebase_file", "get_codebase_commit",
                "get_codebase_tree", "get_codebase_blob",
                "inspect_codebase_symbol",
            } else "web" if name in {
                "search_public_web", "fetch_public_page",
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
        self._append(
            "error",
            "Tool failed",
            error=type(error).__name__,
            message=str(error)[:2000],
        )


@app.get("/health")
async def health() -> dict[str, str]:
    checkpoint_url = os.getenv("CHECKPOINT_DATABASE_URL", "").strip()
    return {
        "status": "ok",
        "checkpointing": (
            "postgresql" if checkpoint_url.startswith(("postgres", "postgresql"))
            else "sqlite" if checkpoint_url.startswith("sqlite")
            else "ephemeral"
        ),
    }


@app.post("/v1/investigations/{execution_id}/cancel")
async def cancel_investigation(
    execution_id: str,
    x_internal_token: str = Header(default=""),
) -> dict[str, str]:
    active = _active_investigations.get(execution_id)
    if active is None:
        return {"status": "not_running"}
    task, expected_token = active
    if not expected_token or not secrets.compare_digest(
        x_internal_token, expected_token
    ):
        raise HTTPException(status_code=403, detail="Invalid internal token")
    task.cancel()
    return {"status": "cancellation_requested"}


@app.post("/v1/investigations")
async def investigate(payload: InvestigationRequest) -> dict[str, Any]:
    current_task = asyncio.current_task()
    execution_id = payload.execution_id or f"direct-{secrets.token_hex(16)}"
    callback_token = (
        payload.callback_token.get_secret_value()
        if payload.callback_token is not None
        else ""
    )
    if payload.execution_id and current_task is not None:
        if execution_id in _active_investigations:
            raise HTTPException(
                status_code=409,
                detail="This investigation execution is already running",
            )
        _active_investigations[execution_id] = (
            current_task,
            callback_token,
        )
    deadline_expired = False

    def enforce_deadline() -> None:
        nonlocal deadline_expired
        deadline_expired = True
        if current_task is not None and not current_task.done():
            current_task.cancel()

    deadline = asyncio.get_running_loop().call_later(
        payload.max_runtime_seconds,
        enforce_deadline,
    )
    try:
        if not payload.database_sources and not payload.codebase_sources:
            raise ValueError("At least one database or codebase source is required")
        sources = [
                {
                    **source.model_dump(exclude={"connection_url"}),
                    "connection_url": source.connection_url.get_secret_value(),
                }
                for source in payload.database_sources
        ]
        codebase_sources = [
            {
                **source.model_dump(exclude={"installation_token"}),
                "installation_token": source.installation_token.get_secret_value(),
            }
            for source in payload.codebase_sources
        ]
        callback = InvestigationTraceCallback(
            payload.database_sources, payload.codebase_sources
        )
        report = None
        evidence: list[Any] = []
        timeline: list[dict[str, Any]] = []
        stage_started_at = _now()
        async for event in stream_investigation(
            payload.question,
            payload.organization_id,
            payload.project_id,
            database_sources=sources,
            codebase_sources=codebase_sources,
            callbacks=[callback],
            investigation_id=execution_id,
            organization_knowledge=payload.organization_knowledge,
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
                                "execution_attempt": payload.execution_attempt,
                                "completed_label": completed_label,
                                "next_stage": current_stage,
                                "next_label": next_label,
                                "started_at": stage_started_at,
                                "completed_at": completed_at,
                                "timestamp": completed_at,
                                "events": callback.events,
                                "evidence": [
                                    _safe(item)
                                    for item in (
                                        update.get("evidence")
                                        if update.get("evidence") is not None
                                        else evidence
                                    )
                                ],
                                "databases": [
                                    {
                                        "connection_id": source.connection_id,
                                        "provider": source.provider,
                                        "analysis_object_count": len(
                                            source.analysis.get("objects", [])
                                        ),
                                    }
                                    for source in payload.database_sources
                                ],
                                "codebases": [
                                    {
                                        "connection_id": source.connection_id,
                                        "provider": source.provider,
                                        "repository": (
                                            f"{source.owner}/{source.repository}"
                                        ),
                                        "branch": source.branch,
                                        "file_count": source.analysis.get(
                                            "file_count", 0
                                        ),
                                    }
                                    for source in payload.codebase_sources
                                ],
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
        # LangGraph may expose state values as Pydantic models or plain dicts,
        # depending on its serialization/checkpoint configuration.
        report_payload = _final_report_payload(report, payload.question)
        response_payload = {
            **report_payload,
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
            "_codebases": [
                {
                    "connection_id": source.connection_id,
                    "provider": source.provider,
                    "repository": f"{source.owner}/{source.repository}",
                    "branch": source.branch,
                    "file_count": source.analysis.get("file_count", 0),
                }
                for source in payload.codebase_sources
            ],
        }
        # Persist the terminal result through a separate short request before
        # returning it on the long-running investigation connection. This
        # prevents a proxy/client disconnect at the end of a multi-minute run
        # from losing an otherwise completed report.
        if payload.completion_url and payload.callback_token:
            try:
                await to_thread(
                    _send_completion,
                    payload.completion_url,
                    payload.callback_token.get_secret_value(),
                    response_payload,
                    payload.execution_attempt,
                )
            except Exception:
                logger.warning(
                    "Could not deliver terminal callback for DeepAgents "
                    "execution %s; the primary response will still be returned.",
                    execution_id,
                    exc_info=True,
                )
        return response_payload
    except asyncio.CancelledError:
        if deadline_expired:
            raise HTTPException(
                status_code=504,
                detail="Investigation exceeded the execution time limit",
            )
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "DeepAgents execution %s failed during %s",
            execution_id,
            (
                timeline[-1]["stage"]
                if "timeline" in locals() and timeline
                else "startup"
            ),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Investigation failed during "
                f"{timeline[-1]['stage'] if 'timeline' in locals() and timeline else 'startup'}: "
                f"{type(exc).__name__}"
            ),
        ) from exc
    finally:
        deadline.cancel()
        await evidence_repository.clear(execution_id)
        if (
            payload.execution_id
            and _active_investigations.get(execution_id, (None, ""))[0]
            is current_task
        ):
            _active_investigations.pop(execution_id, None)
