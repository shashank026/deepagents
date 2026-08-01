from deep_agent.models.query import EvidenceSource, EvidenceSourcePlan
from deep_agent.models.execution import InvestigationPlanStep
from deep_agent.models.state import InvestigationState
from deep_agent.tools.web_research import web_research_enabled


def plan_evidence_sources_node(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    query = state["user_query"].lower()
    sources: list[EvidenceSource] = []
    reasons: dict[EvidenceSource, str] = {}

    source_inventory_known = (
        "database_sources" in state or "codebase_sources" in state
    )
    has_database = bool(state.get("database_sources"))
    has_codebase = bool(state.get("codebase_sources"))
    semantic_terms = {
        "success", "successful", "failed", "failure", "pending", "status",
    }
    code_terms = {
        "mapping", "enum", "implementation", "code", "logic", "configured",
        "configuration", "source code", "where is",
    }
    explicit_code = any(term in query for term in code_terms)
    explicit_logs = any(term in query for term in {"logs", "log entries", "search logs", "show logs"})
    incident = understanding.intent == "incident_investigation"
    has_concrete_identifiers = any(
        entity.value is not None for entity in understanding.entities
    )

    if not source_inventory_known:
        has_database = not (
            (explicit_code or explicit_logs)
            and not incident
            and not has_concrete_identifiers
        )
        has_codebase = bool(
            incident
            or semantic_terms.intersection(understanding.business_terms)
            or explicit_code
        )

    # Explicit source lookups can be source-only; business data and incidents
    # retain database evidence as the factual system-of-record baseline.
    if has_database:
        sources.append(EvidenceSource.DATABASE)
        reasons[EvidenceSource.DATABASE] = (
            "Resolve project-scoped entities, schemas, relationships, stored "
            "values, and runtime state."
        )

    # A connected repository is an available source, not a mandatory source.
    # Start with the smallest authoritative source set and escalate only when
    # the requested answer actually depends on application behavior.
    needs_code_initially = has_codebase and (
        explicit_code
        or understanding.intent == "explanation"
        or (incident and not has_database)
    )
    if needs_code_initially:
        sources.append(EvidenceSource.CODEBASE)
        reasons[EvidenceSource.CODEBASE] = (
            "Verify application behavior, database relationships, mappings, "
            "validators, and business meaning."
        )

    log_terms = {"error", "exception", "timeout", "incident", "timeline", "logs", "runtime"}
    if incident or explicit_logs or any(term in query for term in log_terms):
        sources.append(EvidenceSource.LOGS)
        reasons[EvidenceSource.LOGS] = "Verify runtime events, failures, and event chronology."

    explicit_external = any(term in query for term in {
        "documentation", "specification", "release note", "security advisory",
        "official docs", "vendor docs", "upstream",
    })
    if web_research_enabled() and explicit_external:
        sources.append(EvidenceSource.WEB)
        reasons[EvidenceSource.WEB] = (
            "Consult allowlisted official public documentation for current "
            "provider behavior, specifications, limits, error semantics, release "
            "changes, and recommended operational guidance. External material "
            "is context only for customer-specific incidents."
        )

    # Preserve order while removing duplicates introduced by overlapping rules.
    sources = list(dict.fromkeys(sources))
    optional_sources: list[EvidenceSource] = []
    escalation_reasons: dict[EvidenceSource, list[str]] = {}
    if has_codebase and EvidenceSource.CODEBASE not in sources:
        optional_sources.append(EvidenceSource.CODEBASE)
        escalation_reasons[EvidenceSource.CODEBASE] = [
            "The stored value has unresolved business meaning.",
            "Database or runtime evidence contradicts expected behavior.",
            "The implementing decision path is required to explain root cause.",
        ]
    if EvidenceSource.LOGS not in sources and incident:
        optional_sources.append(EvidenceSource.LOGS)
        escalation_reasons[EvidenceSource.LOGS] = [
            "A runtime chronology or error event is required to test a hypothesis."
        ]
    if (
        web_research_enabled()
        and EvidenceSource.WEB not in sources
        and understanding.intent in {
            "incident_investigation", "informational", "explanation",
        }
    ):
        optional_sources.append(EvidenceSource.WEB)
        escalation_reasons[EvidenceSource.WEB] = [
            "An unresolved public provider behavior, specification, release change, or advisory requires an official source."
        ]
    plan = EvidenceSourcePlan(
        sources=sources,
        reasons=reasons,
        optional_sources=optional_sources,
        escalation_reasons=escalation_reasons,
    )
    investigation_plan = []
    if EvidenceSource.CODEBASE in sources:
        investigation_plan.append(InvestigationPlanStep(
            stage="codebase",
            objective=(
                "Trace the failing entry point through every called service, "
                "repository, validator, model, constant, and feature mapping "
                "until the exact decision predicate is established."
            ),
            completion_criteria=(
                "Focused code evidence identifies the authoritative entry "
                "point, decision function, and relevant fields/constants."
            ),
        ))
    if EvidenceSource.DATABASE in sources:
        investigation_plan.extend([
            InvestigationPlanStep(
                stage="schema",
                objective="Discover relevant objects, fields, indexes, relationships, and values.",
                completion_criteria=(
                    "Schema evidence identifies every object and field needed "
                    "for the requested answer or causal predicate."
                ),
            ),
            InvestigationPlanStep(
                stage="database",
                objective=(
                    "Use relationships and predicates proven by code/schema to "
                    "query the affected entity and the runtime state selected "
                    "by the decision function."
                ),
                completion_criteria=(
                    "Successful query evidence verifies the affected entity and "
                    "the current source-of-truth state."
                ),
            ),
        ])
    if EvidenceSource.LOGS in sources:
        investigation_plan.append(InvestigationPlanStep(
            stage="logs",
            objective="Find runtime errors and establish the incident timeline.",
            completion_criteria=(
                "Relevant log evidence is collected, or the source is explicitly unavailable."
            ),
        ))
    if EvidenceSource.WEB in sources:
        investigation_plan.append(InvestigationPlanStep(
            stage="web",
            objective=(
                "Find current official public documentation relevant to the "
                "question and preserve its title and URL."
            ),
            completion_criteria=(
                "Relevant official documentation is cited, or public research "
                "is explicitly unavailable."
            ),
        ))
    investigation_plan.append(InvestigationPlanStep(
        stage="validation",
        objective="Cross-check evidence, reject alternatives, and validate the report.",
        completion_criteria=(
            "Every conclusion cites evidence and all material contradictions "
            "or unavailable sources are recorded."
        ),
    ))
    return {
        "evidence_source_plan": plan,
        "investigation_plan": investigation_plan,
        "current_stage": "evidence_collection",
    }
