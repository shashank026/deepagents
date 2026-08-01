from deep_agent.models.harness import (
    InvestigationContextManifest,
    InvestigationLimits,
    SourceCapability,
)
from deep_agent.models.state import InvestigationState
from deep_agent.tools.external_sources import log_source_status


def build_context_manifest_node(state: InvestigationState) -> dict:
    """Describe connected authority without exposing connection URLs or tokens."""
    sources: list[SourceCapability] = []
    for source in state.get("database_sources", []):
        analysis = source.get("analysis", {})
        sources.append(SourceCapability(
            source_type="database",
            source_id=str(source.get("connection_id", "unknown")),
            provider=source.get("provider") or analysis.get("database_type"),
            environment=source.get("environment"),
            version=analysis.get("schema_hash"),
            metadata={"object_count": len(analysis.get("objects", []))},
        ))
    for source in state.get("codebase_sources", []):
        analysis = source.get("analysis", {})
        sources.append(SourceCapability(
            source_type="codebase",
            source_id=str(source.get("connection_id", "unknown")),
            provider=source.get("provider") or "github",
            version=source.get("commit_sha") or source.get("branch"),
            metadata={
                "repository": source.get("repository"),
                "file_count": analysis.get("file_count", 0),
            },
        ))
    logs = log_source_status()
    if logs["available"]:
        sources.append(SourceCapability(
            source_type="logs",
            source_id="configured-log-source",
            provider=logs.get("provider"),
        ))
    manifest = InvestigationContextManifest(
        investigation_id=state["investigation_id"],
        organization_id=state["organization_id"],
        project_id=state["project_id"],
        sources=sources,
        limits=InvestigationLimits(
            deadline_seconds=state.get("max_runtime_seconds"),
        ),
        unavailable_sources=[
            name for name, present in {
                "database": bool(state.get("database_sources")),
                "codebase": bool(state.get("codebase_sources")),
                "logs": bool(logs["available"]),
            }.items() if not present
        ],
    )
    return {
        "context_manifest": manifest.model_dump(mode="json"),
        "current_stage": "query_understanding",
    }
