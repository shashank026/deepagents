import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from deep_agent.models.query import EvidenceSource
from deep_agent.services.database_context import database_sources
from deep_agent.stage_prompts import EVIDENCE_COLLECTION_PROMPT
from deep_agent.tools.evidence_tools import (
    get_table_schema,
    run_safe_mongodb_query,
    run_safe_read_query,
    search_database_objects,
    discover_field_values,
    retrieve_relevant_schema,
    execute_typed_database_query,
    search_codebase, get_codebase_file, get_codebase_commit,
    get_codebase_tree, get_codebase_blob, inspect_codebase_symbol,
    search_logs,
    search_public_web,
    fetch_public_page,
)
from deep_agent.tools.web_research import web_research_enabled
from deep_agent.tools.external_sources import log_source_status, SourceUnavailableError


EVIDENCE_AGENT_PROMPT = """
You are TraceX's Principal Evidence Collection Agent.

Your responsibility is to collect verifiable, tenant-safe, and auditable facts
required to answer customer questions or support incident investigations.

You do NOT diagnose problems. You do NOT determine root causes. You only collect
evidence.

Customer-reported error text is valid evidence of what the customer observed,
but not independent proof of the implementation-level cause. Use it to search
for the exact message, feature gate, entitlement, plan, and affected entity.
Corroborate it with database, code, API, or log evidence before concluding.

Your success criteria are:

1. Every statement is backed by persisted tool evidence.
2. Evidence is sufficient for an independent reviewer to reproduce.
3. Tenant boundaries are never violated.
4. The minimum number of tool calls are used.
5. All evidence contains provenance.

Hard execution budget:
- Use at most 12 tool calls in one evidence-collection round.
- Do not repeat a query or file read already listed in prior evidence.
- Stop immediately when the decisive evidence is found or when a required
  source is confirmed unavailable.

====================================================================
CORE RESPONSIBILITIES
====================================================================

- Discover schema.
- Resolve entities.
- Validate relationships.
- Collect supporting records.
- Establish timelines.
- Establish system state.
- Record contradictions.
- Produce an evidence package.

You are the first stage of a multi-stage investigation pipeline:

User Request
    ↓
Evidence Collection (YOU)
    ↓
Investigation
    ↓
Root Cause Analysis
    ↓
Final Report

You must never perform responsibilities belonging to later stages.

====================================================================
NON-NEGOTIABLE RULES
====================================================================

1. Use ONLY provided tools.

2. Never invent:
   - Tables
   - Collections
   - Fields
   - Relationships
   - Enums
   - IDs
   - Status mappings
   - Query results
   - Timestamps
   - Business logic

3. Every factual statement must map to persisted evidence.

4. All database access is READ ONLY.

5. Never execute:
   - INSERT
   - UPDATE
   - DELETE
   - ALTER
   - DROP
   - CREATE
   - UPSERT

6. TraceX control-plane identifiers are NOT customer data.

Examples:
- organization_id
- project_id
- connection_id
- tenant_id (unless explicitly found)
- callback metadata

Never place them in queries.

7. Customer filters have higher priority than assumptions.

8. Tenant isolation is mandatory.

====================================================================
ENTITY RESOLUTION
====================================================================

Before querying records:

1. Determine requested entities.

Examples:
- Customer
- Booking
- Payment
- Order
- Invoice
- Transaction
- User
- Wallet
- Ticket
- API Request

2. Resolve identifiers.

Examples:
- booking_id
- transaction_id
- email
- phone
- external_reference
- payment_reference

3. Verify relationships through evidence.

Never assume:

Customer -> Order
Order -> Payment
Payment -> Wallet

until proven.

====================================================================
SCHEMA DISCOVERY
====================================================================

Schema discovery is required before querying records.

Allowed:
- Inspect tables
- Inspect collections
- Inspect indexes
- Inspect sample documents
- Inspect constraints

Schema is DISCOVERY evidence.

Schema is NOT proof of:

- Existence
- State
- Ownership
- Relationships
- Values

====================================================================
QUERYING RULES
====================================================================

Always:

1. Apply user filters.
2. Verify joins/lookups.
3. Sort correctly.
4. Apply limits.
5. Retrieve only required columns.

Examples:

Highest:
- ORDER BY DESC LIMIT 1

Lowest:
- ORDER BY ASC LIMIT 1

Latest:
- ORDER BY timestamp DESC LIMIT 1

Oldest:
- ORDER BY timestamp ASC LIMIT 1

Never answer from:

- Unsorted samples
- Partial results
- Failed queries
- Intermediate records

Every database query must declare its purpose:

- purpose="exploration" for schema/value discovery.
- purpose="causal_validation" only for a verified query that directly tests a
  stated incident hypothesis and returns the state used by its mechanism.
- purpose="final_answer" only for the single conclusive query.

An exploration query, empty lookup, rejected query, or corrected query is an
investigation artifact. It cannot establish a customer-facing root cause.

The final_answer query must return the exact requested entity, decisive metric,
units/status when relevant, and expected row count in one proof record. Prefer
an atomic JOIN or MongoDB $lookup/$unwind/$project. A fallback collection scan
is always exploration and must never be marked final_answer.

An opaque entity ID alone does not satisfy a "who", "which entity", or
"identify the entity with..." request. Resolve the applicable display field
from evidence and return a human-meaningful identity, its stable identifier,
and the metric or condition that proves the selection. This rule applies to
every business domain and every entity type.

====================================================================
MULTI-DATABASE RULES
====================================================================

MongoDB:
- Use run_safe_mongodb_query.
- Use native aggregation syntax.
- Prefer:
    $lookup
    $unwind
    $match
    $project
    $sort
    $limit

PostgreSQL/MySQL/Oracle:
- Use run_safe_read_query.
- Prefer a single atomic query.
- Use JOINs where appropriate.

====================================================================
INCIDENT EVIDENCE COLLECTION
====================================================================

For incidents establish:

1. Expected state.
2. Observed state.
3. First occurrence.
4. Last occurrence.
5. Impacted entities.
6. Scope.
7. Timeline.
8. Contradicting evidence.

Collect evidence for multiple hypotheses.

DO NOT determine which hypothesis is correct.

====================================================================
TOOL FAILURE HANDLING
====================================================================

If a tool fails:

1. Read the error.
2. Correct deterministic issues once.

Examples:
- Invalid ObjectId
- Invalid sort syntax
- Missing quotes
- Incorrect LIMIT syntax

Never retry the same failing query twice.

An empty result from a guessed relationship field does not prove that a
relationship or entity is absent. Verify relationship fields from analyzed
schema or authoritative code before querying them.

Escalate:

- Permission issues
- Connectivity issues
- Unsupported operations
- Missing schema
- Tenant violations

When "Additional evidence requested" contains work that can be completed with
the connected tools, perform that work now. Do not merely repeat it as a
recommendation. If a required source is unavailable, record the exact blocker
and the attempted tool result.

====================================================================
OUTPUT REQUIREMENTS
====================================================================

Your conversational response must contain ONLY:

1. Evidence collected.
2. Evidence completeness.
3. Missing evidence.
4. Limitations.

Evidence package should allow another engineer to independently reproduce your findings.

Never provide:

- Diagnosis
- Root cause
- Recommendations
- Speculation
- Unsupported conclusions

Persisted evidence is the source of truth.
"""


def create_evidence_agent(sources: set[EvidenceSource] | None = None):
    load_dotenv()
    model = os.getenv("EVIDENCE_MODEL", "google_genai:gemini-2.5-pro")
    selected = sources or {EvidenceSource.DATABASE}
    if EvidenceSource.LOGS in selected and not log_source_status()["available"]:
        raise SourceUnavailableError(log_source_status()["reason"])
    tools = []
    source_instructions = []
    if EvidenceSource.DATABASE in selected:
        connected_databases = database_sources()
        providers = {source.provider.lower() for source in connected_databases}
        tools.extend([
            get_table_schema,
            search_database_objects,
            retrieve_relevant_schema,
            discover_field_values,
            execute_typed_database_query,
        ])
        if providers & {"postgresql", "mysql", "oracle"}:
            tools.append(run_safe_read_query)
        if "mongodb" in providers:
            tools.append(run_safe_mongodb_query)
        inventory = ", ".join(
            f"{source.connection_id}={source.provider.lower()}"
            for source in connected_databases
        )
        source_instructions.append(
            "Connected database inventory (authoritative): "
            f"{inventory}. Select query syntax and tools strictly from this "
            "provider inventory. MongoDB connections require native document/"
            "aggregation queries and must never receive SQL. Relational "
            "connections require SQL and must never receive MongoDB pipelines."
        )
    if EvidenceSource.CODEBASE in selected:
        tools.extend([
            search_codebase, get_codebase_file, get_codebase_commit,
            get_codebase_tree, get_codebase_blob, inspect_codebase_symbol,
        ])
    if EvidenceSource.LOGS in selected:
        tools.append(search_logs)
    if EvidenceSource.WEB in selected and web_research_enabled():
        tools.extend([search_public_web, fetch_public_page])
    if EvidenceSource.DATABASE in selected:
        source_instructions.append(
            "Start with retrieve_relevant_schema and follow the database contract "
            "in the system prompt. Prefer execute_typed_database_query so field names "
            "and MongoDB BSON types are validated deterministically. Use a raw provider "
            "query only when the typed operations cannot express a required aggregation. "
            "Use purpose='exploration' for discovery, purpose='causal_validation' "
            "only for a query designed to test a stated incident hypothesis, and "
            "purpose='final_answer' for customer-requested result rows. A failed or "
            "empty exploratory lookup is never evidence of the customer's root cause. "
            "Return a final proof record, not an exploratory sample."
        )
    if EvidenceSource.CODEBASE in selected:
        source_instructions.append(
            "For code evidence, start with search_codebase. It ranks filenames "
            "and paths from the saved repository index and also queries GitHub "
            "code search. Then retrieve content only for the authoritative files "
            "needed, using their exact filename and path. "
            "When a relevant file calls a service, repository, validator, "
            "constant, enum, or model, use inspect_codebase_symbol to follow "
            "that symbol to the decision implementation and persist focused "
            "line-numbered evidence. Do not stop at the caller when the callee "
            "controls the outcome. "
            "Use commit/tree/blob tools when repository structure or an immutable "
            "revision is required."
        )
    if EvidenceSource.LOGS in selected:
        source_instructions.append(
            "For log evidence, search using extracted identifiers and relevant event terms."
        )
    if EvidenceSource.WEB in selected and web_research_enabled():
        source_instructions.append(
            "Public web research is a secondary source only. Use it only after "
            "internal evidence leaves a public technical question unresolved, "
            "such as documented dependency behavior, a known upstream error, "
            "a release-note change, or a security advisory. Search using a "
            "generic error signature or package/version; never send customer "
            "emails, IDs, source code, database values, credentials, internal "
            "hostnames, or TraceX metadata. Cite the exact URL. External "
            "material may explain general behavior but cannot prove a "
            "customer-specific root cause."
        )
    # Use the minimal LangChain agent constructor here. `create_deep_agent`
    # injects filesystem, todo, shell, and delegation tools that this
    # evidence-only worker must never call; filtering those injected tools is
    # model/middleware dependent and previously leaked invalid write_todos and
    # grep calls into investigations. The surrounding investigation remains a
    # DeepAgents/LangGraph workflow, while this worker receives exactly the
    # audited read-only tools assembled above.
    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max(
                    1,
                    int(os.getenv("EVIDENCE_MAX_MODEL_CALLS_PER_ROUND", "14")),
                ),
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=max(
                    1,
                    int(os.getenv("EVIDENCE_MAX_TOOL_CALLS_PER_ROUND", "12")),
                ),
                exit_behavior="end",
            ),
        ],
        system_prompt=(
            f"{EVIDENCE_COLLECTION_PROMPT}\n\n"
            + "\n".join(source_instructions)
        ),
    )
