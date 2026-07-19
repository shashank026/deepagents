import argparse
import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console

from deep_agent.models.report import RootCauseReport
from deep_agent.services.progress import bind_progress_sink, reset_progress_sink
from deep_agent.workflow.investigation_graph import create_investigation_graph

STAGE_MESSAGES = {
    "collect_evidence": "Collecting evidence",
    "validate_evidence": "Validating collected evidence",
    "investigate": "Comparing expected and observed state",
    "identify_root_cause": "Evaluating root-cause hypotheses",
    "validate_root_cause": "Validating the conclusion",
    "build_final_report": "Preparing the final report",
}

NEXT_STAGE_MESSAGES = {
    "evidence_validation": "Validating collected evidence",
    "investigation": "Comparing expected and observed state",
    "root_cause_analysis": "Evaluating root-cause hypotheses",
    "root_cause_validation": "Validating the conclusion",
    "final_report": "Preparing the final report",
    "inconclusive": "Preparing an inconclusive report",
}


class WorkflowProgressCallback(BaseCallbackHandler):
    """Render live model/tool activity to stderr while stdout stays valid JSON."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.console.print("  [dim]Model is evaluating the next action…[/]")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name") or "tool"
        self.console.print(f"  [cyan]↳[/] Running [bold]{name}[/]")

    def on_tool_end(self, output, **kwargs) -> None:
        self.console.print("  [green]↳ Tool completed[/]")

    def on_tool_error(self, error, **kwargs) -> None:
        self.console.print(f"  [red]↳ Tool failed:[/] {error}")


def initial_state(user_query: str, organization_id: str, project_id: str) -> dict:
    return {
        "investigation_id": str(uuid4()), "user_query": user_query,
        "organization_id": organization_id, "project_id": project_id,
        "extracted_entities": {}, "evidence": [], "evidence_collection_attempts": 0,
        "evidence_collection_errors": [], "requested_evidence": [],
        "investigation": None, "root_cause_analysis": None, "final_report": None,
        "current_stage": "starting", "failure_reason": None,
    }


async def investigate_issue(user_query: str, organization_id: str = "local",
                            project_id: str = "default") -> RootCauseReport:
    state = initial_state(user_query, organization_id, project_id)
    final = await create_investigation_graph().ainvoke(
        state, config={"recursion_limit": 20,
                       "configurable": {"thread_id": state["investigation_id"]}},
    )
    return final["final_report"]


async def stream_investigation(user_query: str, organization_id: str = "local",
                               project_id: str = "default",
                               callbacks: list | None = None) -> AsyncIterator[dict]:
    state = initial_state(user_query, organization_id, project_id)
    graph = create_investigation_graph()
    async for event in graph.astream(
        state, config={"recursion_limit": 20, "callbacks": callbacks or [],
                       "configurable": {"thread_id": state["investigation_id"]}},
        stream_mode="updates",
    ):
        yield event


async def _run_cli(question: str, organization_id: str, project_id: str,
                   show_progress: bool) -> RootCauseReport:
    console = Console(stderr=True)
    callbacks = [WorkflowProgressCallback(console)] if show_progress else []
    progress_token = bind_progress_sink(
        lambda message: console.print(f"[yellow]⏳ {message}[/]")
        if show_progress else None
    )
    report: RootCauseReport | None = None
    announced: str | None = None
    if show_progress:
        announced = "collect_evidence"
        console.print(f"[cyan]→ {STAGE_MESSAGES[announced]}[/]")
    try:
        async for event in stream_investigation(
            question, organization_id, project_id, callbacks=callbacks
        ):
            for node_name, update in event.items():
                if show_progress:
                    console.print(f"[green]✓ {STAGE_MESSAGES.get(node_name, node_name)}[/]")
                if update.get("final_report") is not None:
                    report = update["final_report"]
                current_stage = update.get("current_stage")
                next_message = NEXT_STAGE_MESSAGES.get(current_stage)
                if show_progress and next_message and current_stage != announced:
                    announced = current_stage
                    console.print(f"[cyan]→ {next_message}[/]")
    finally:
        reset_progress_sink(progress_token)
    if report is None:
        raise RuntimeError("Workflow completed without a final report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-based database investigation")
    parser.add_argument("question")
    parser.add_argument("--organization-id", default="local")
    parser.add_argument("--project-id", default="default")
    parser.add_argument("--quiet", action="store_true", help="Print only report JSON")
    args = parser.parse_args()
    report = asyncio.run(_run_cli(
        args.question, args.organization_id, args.project_id, not args.quiet
    ))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
