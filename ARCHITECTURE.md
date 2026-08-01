# TraceX investigation architecture

## Diagnosis

The original implementation had a sound LangGraph shell, but reliability was
limited by:

1. A large evidence-agent prompt carrying planning, safety, query, and report
   responsibilities simultaneously.
2. Direct `with_structured_output` usage with no fallback when a provider
   rejected `response_format` or JSON Schema.
3. Evidence retries controlled mainly by model fields rather than recorded
   failed assumptions and deterministic budgets.
4. Repeated evidence being sent to later models, increasing latency and token
   usage.
5. No final deterministic report gate for evidence references and secrets.
6. No dedicated semantic-value discovery tool for encoded statuses and enums.

## Production topology

TraceX uses one deterministic orchestrator, one adaptive evidence agent, and
one shared reasoning service:

```text
API / task worker
  -> LangGraph orchestrator (owns state, routing, deadlines, retries)
     -> Evidence agent (read-only tools, hard model/tool-call limits)
     -> Shared reasoning service (investigation + RCA typed outputs)
     -> Deterministic validators (evidence, self-check, RCA, report)
```

This is intentionally not a swarm. Each responsibility has one owner:

- LangGraph is the only orchestrator.
- The evidence agent is the only component allowed to choose retrieval tools.
- The shared reasoning service owns one model client, concurrency control,
  timeouts, structured-output repair, and process-wide model configuration.
- Python validators—not an LLM—decide whether evidence references, tenant
  boundaries, confidence thresholds, root cause, and customer-safe output pass.

Every request has a stable execution ID. The API rejects duplicate active
executions, propagates the ID through LangGraph and evidence storage, supports
cancellation/deadlines, and releases execution-scoped evidence in `finally`.
For multi-instance deployment, run the API behind an external durable task
queue and persist terminal workflow/callback state in `backend_base`; no
process-local active-task registry should be treated as a distributed lock.

## Corrected design

LangGraph owns the lifecycle. DeepAgents remains a bounded, read-only evidence
executor because repository/schema exploration requires adaptive tool choice.

```text
intake/extract
  -> plan sources and investigation steps
  -> collect evidence with bounded DeepAgents
  -> validate evidence
  -> analyze evidence and generate hypotheses
  -> deterministic self-check
       -> revise plan -> collect evidence (bounded)
       -> determine root cause
       -> build inconclusive/retrieval report
  -> validate root cause
  -> build report
  -> deterministic report validation/redaction
  -> END
```

Deterministic Python owns tenant isolation, query validation, retry ceilings,
evidence-ID validation, failed-assumption tracking, report repair, and routing.

## Expert investigation harness

TraceX starts with the minimum authoritative source set. A connected repository
is a capability, not a requirement: ordinary record retrieval starts with the
database, while code is progressively added only when implementation behavior,
an unresolved mapping, a contradiction, or a causal decision path requires it.
The source plan records optional sources and their generic escalation reasons.

Every run begins with a secret-free context manifest containing tenant/project
identity, connected source capabilities, providers, schema/repository versions,
permissions, and limits. Connection URLs and tokens are never included.

Database agents prefer a provider-neutral `TypedQueryIntent`. Deterministic
compilers verify objects and fields against analyzed metadata before producing
MongoDB operations or read-only SQL. MongoDB filters are coerced from analyzed
BSON types rather than field-name conventions; this handles an `ObjectId` field
named `objectType`, for example. Mixed analyzed types cause an explicit value-
discovery requirement instead of a guess. The guarded raw query tools remain
available for aggregations not yet expressible by the typed operations, and
apply the same MongoDB schema-aware coercion.

Investigation skills live under `skills/*/SKILL.md`. Only skills matching the
typed question intent and planned sources are loaded into the evidence prompt.
Skills contain generic workflows and evidence standards, never customer or
industry facts. Verified organization/project investigation memory continues
to be supplied as planning hints only and must be revalidated against current
evidence.

Production durability is implemented through two separate stores. LangGraph
uses a configured async PostgreSQL checkpointer with synchronous step
durability; SQLite is supported for durable local development. The backend
persists each bounded evidence item in an append-only tenant/project-scoped
ledger with a content hash. A resumed process rehydrates its execution-local
evidence cache from checkpointed state.

Multiple backend instances coordinate through database execution leases. A
dispatcher claims queued or expired work under configurable global,
organization, and project limits; heartbeats renew ownership and an execution-
attempt fencing token rejects late progress, failure, or completion callbacks
from an expired worker. Independent planned sources are collected concurrently
by bounded source-specific workers, while the parent LangGraph remains the only
orchestrator and deterministic validator.

Source relevance and source availability are separate decisions. In
particular, incident intent does not create a logs capability. Logs are planned
and the `search_logs` tool is exposed only when `LOG_ROOT` resolves to an
existing authorized directory (or a future connected logs adapter reports an
equivalent capability). Helpful but unavailable sources are recorded once in
the source plan. Permanent configuration, authentication, authorization, and
source-unavailable failures are not retried; independent successful workers
continue and the final report identifies the unavailable evidence.

Sandboxed code execution is intentionally not part of this rollout. Future
sandbox workers must be isolated, network-denied by default, and must never
receive production database credentials.

### Decision-path tracing

Incident evidence collection follows a code-to-runtime causal path:

1. Resolve the affected business entity.
2. Locate the failing entry point or reported error.
3. Follow called services, repositories, validators, models, constants, and
   enums with `inspect_codebase_symbol`.
4. Extract the database relationship and selection predicate from code.
5. Query the runtime record selected by that predicate.
6. Compare stored state with the authoritative status or feature mapping.
7. Persist the causal chain and reject alternatives before RCA.

Focused symbol evidence retains repository, ref, path, blob SHA, line numbers,
and bounded source snippets, so decisive logic is not lost when large files are
compacted.

### Organization-independent research

The workflow receives only the database and codebase sources already scoped by
`organization_id` and `project_id` in `backend_base`. It never discovers or
selects sources belonging to another organization.

### Organization-scoped adaptive memory

Before a new investigation starts, `backend_base` performs vectorless lexical
retrieval over recent completed investigations from the exact same
organization and project. Only outcomes whose deterministic verification
status is `verified` are eligible. Selected summaries are redacted and sent to
DeepAgents as planning hints.

Prior learnings are never treated as evidence. They may prioritize a likely
schema, code path, provider concept, or failed assumption, but the current
investigation must revalidate every claim against the currently connected
database, codebase, logs, or official documentation. This provides safe online
adaptation without model-weight mutation, cross-tenant training, or automatic
reinforcement of unverified answers.

For every project query, the planner uses every connected source type:

- code establishes behavior, mappings, validators, and relationships;
- schema establishes available objects and fields;
- bounded read-only queries establish current runtime state;
- logs establish runtime chronology when configured and relevant.

The plan records completion criteria and evidence IDs for each source. Missing
coverage triggers a targeted additional collection round, up to the configured
ceiling. Verified explanatory questions produce a resolved explanation report;
data retrieval produces a verified result report; incidents require a supported
causal mechanism before an RCA is marked resolved.

## Important files

```text
src/deep_agent/
├── stage_prompts.py                 # compact shared policy and stage prompts
├── models/
│   ├── evidence.py                  # persisted evidence
│   ├── execution.py                 # capabilities, errors, assumptions, plan
│   ├── investigation.py             # hypotheses and investigation result
│   ├── report.py                    # final structured report
│   └── state.py                     # typed LangGraph state
├── services/
│   ├── structured_output.py         # native output + JSON/Pydantic repair
│   ├── reasoning.py                 # shared model, semaphore, timeout, budget
│   ├── evidence_repository.py       # evidence persistence interface/store
│   └── evidence_context.py          # bounded model context
├── nodes/
│   ├── source_planning.py
│   ├── evidence_collection.py
│   ├── evidence_validation.py
│   ├── investigation.py
│   ├── self_check.py
│   ├── root_cause.py
│   ├── root_cause_validation.py
│   ├── report_builder.py
│   └── report_validation.py
├── tools/
│   ├── database.py                  # provider-specific safe execution
│   ├── evidence_tools.py            # persisted evidence contracts
│   └── github.py                    # focused GitHub reads/search
└── workflow/investigation_graph.py
```

## Structured-output compatibility

`MODEL_STRUCTURED_OUTPUT_MODE=auto` attempts the provider's native structured
output. Capability errors fall back to JSON-only generation, safe JSON
extraction, Pydantic validation, and at most `MAX_OUTPUT_REPAIRS` repair calls.
Set the mode to `json` to force the portable path.

## Operational limits

The following settings are enforced in code:

- `REASONING_MAX_CONCURRENCY`: simultaneous investigation/RCA model calls.
- `REASONING_TIMEOUT_SECONDS`: deadline for one reasoning stage.
- `REASONING_MAX_CALLS_PER_INVESTIGATION`: total analytical call budget.
- `EVIDENCE_MAX_MODEL_CALLS_PER_ROUND`: evidence-agent model call budget.
- `EVIDENCE_MAX_TOOL_CALLS_PER_ROUND`: evidence-agent tool call budget.
- `MAX_EVIDENCE_ITEMS_PER_INVESTIGATION`: bounded execution-scoped storage.
- Request `max_runtime_seconds`: end-to-end deadline enforced by the API.

Model payloads and credentials are not logged. Stage, model name, and elapsed
time are logged for capacity planning and incident diagnosis.

## Migration

1. Pull/install the updated DeepAgents package with `uv sync`.
2. Copy new non-secret settings from `.env.example`.
3. Restart DeepAgents and `backend_base`.
4. Re-sync MongoDB connections once so field/index/relationship metadata is
   regenerated by the fixed sampler.
5. Existing completed investigations remain immutable; run a new investigation.

No database migration is required for these workflow-state changes because
workflow state is execution-scoped and final JSON remains backward compatible.

## Run and test

```bash
cd /Users/shashank/Documents/TraceX/deepagents
uv sync
uv run uvicorn deep_agent.api:app --app-dir src --host 0.0.0.0 --port 8010
uv run pytest tests -q
```

## Example

Request:

```text
Campaign creation for project Future-9798 returns:
"Your current plan does not include this campaign type."
```

Expected report behavior:

- Persist the customer message as medium-reliability user-input evidence.
- Inspect campaign entitlement code, project schema, subscription, and plan.
- Establish a root cause only if independent evidence corroborates the rejected
  entitlement.
- Otherwise return `insufficient_evidence` with the exact missing mapping,
  runtime event, or repository source.

Example report shape:

```json
{
  "investigation_status": "probable_root_cause",
  "root_cause": "The requested campaign type is not enabled by the project's current plan.",
  "confidence": 0.82,
  "supporting_evidence_ids": ["ev-user-input-...", "ev-subscription", "ev-code-gate"],
  "expected_state": "The selected campaign type is enabled for the project plan.",
  "actual_state": "The entitlement check rejected the campaign type.",
  "rejected_hypotheses": ["User account is inactive."],
  "missing_information": []
}
```

The exact conclusion and confidence must be generated from retrieved evidence;
the example is a schema/behavior illustration, not a hard-coded result.
