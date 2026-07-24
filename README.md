# Deterministic RCA workflow with LangGraph and DeepAgents

This project investigates database-backed incidents through a parent LangGraph.
The graph—not an agent prompt—enforces the lifecycle:

```text
START → extract business entities → plan evidence sources → collect evidence
      → validate evidence → investigate
      → identify root cause → validate root cause → build report → END
```

The evidence stage uses a DeepAgent because schema discovery and read-only query
selection require dynamic exploration. Evidence validation, retry routing,
root-cause confidence checks, and report construction are ordinary Python.

## Safety and traceability

- Evidence tools persist a typed `Evidence` record for every schema inspection
  and database query. The agent's final message is not the evidence store.
- PostgreSQL, MySQL, and Oracle SQL is restricted to one read-only statement,
  capped at 100 rows, and limited to 15 seconds. MongoDB uses a separate,
  structured find/aggregation tool that rejects write stages and server-side
  code and caps results at 100 documents.
- Investigation and root-cause stages use Pydantic structured output.
- Model-produced evidence and hypothesis IDs are checked against workflow state.
- Evidence collection loops are capped at three attempts.
- Gemini `429 RESOURCE_EXHAUSTED` responses honor the provider retry delay with
  jitter; `MODEL_RATE_LIMIT_RETRIES` controls the retry count (default: `2`).
- Project database analyses supplied at request time are searched locally with lexical ranking, stemming,
  synonyms, and partition-family diversity. No embedding service or vector
  database is required.
- Evidence-source planning selects the database, codebase, logs, or a combination
  before the DeepAgent is created. Set `CODEBASE_ROOT` and `LOG_ROOT` to the
  authorized application source and log directories.
- Reports are assembled deterministically and expose database rows as typed
  result records.

## Setup

Install [uv](https://docs.astral.sh/uv/) and synchronize the locked environment:

```bash
uv sync
```

Configure at least:

```dotenv
# Database connection URLs and analysis metadata are supplied per investigation
# by backend_base. Do not configure a target DATABASE_URL here.
GOOGLE_API_KEY=...
```

Optional model overrides are `EVIDENCE_MODEL`, `INVESTIGATION_MODEL`, and
`ROOT_CAUSE_MODEL`.

## Run

```bash
uv run uvicorn deep_agent.api:app --reload --host 127.0.0.1 --port 8010
```

`backend_base` calls this internal service and supplies the authenticated
project's latest database analyses and connection URLs for each investigation.
The API is not called by the browser and does not return connection secrets.

Application code can call `deep_agent.main.investigate_issue` for a final
`RootCauseReport`, or `stream_investigation` for LangGraph node updates, when it
also supplies the runtime database sources.

## Layout

```text
src/deep_agent/
├── agents/       # focused evidence DeepAgent
├── models/       # evidence, hypotheses, RCA, report, shared state
├── nodes/        # bounded workflow stages and routing
├── services/     # evidence repository and invocation context
├── tools/        # structured schema and safe database tools
└── workflow/     # parent LangGraph assembly
```

Run the local checks with `uv run pytest`. The test suite does not call the
database or external models. Use `uv lock --upgrade` when intentionally updating
dependencies, and commit both `pyproject.toml` and `uv.lock`.
