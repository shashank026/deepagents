# Deterministic RCA workflow with LangGraph and DeepAgents

This project investigates database-backed incidents through a parent LangGraph.
The graph—not an agent prompt—enforces the lifecycle:

```text
START → collect evidence → validate evidence → investigate
      → identify root cause → validate root cause → build report → END
```

The evidence stage uses a DeepAgent because schema discovery and read-only query
selection require dynamic exploration. Evidence validation, retry routing,
root-cause confidence checks, and report construction are ordinary Python.

## Safety and traceability

- Evidence tools persist a typed `Evidence` record for every schema inspection
  and database query. The agent's final message is not the evidence store.
- SQL is restricted to one read-only statement, capped at 100 rows, executed in
  a read-only PostgreSQL transaction, and limited to 15 seconds.
- Investigation and root-cause stages use Pydantic structured output.
- Model-produced evidence and hypothesis IDs are checked against workflow state.
- Evidence collection loops are capped at three attempts.
- Gemini `429 RESOURCE_EXHAUSTED` responses honor the provider retry delay with
  jitter; `MODEL_RATE_LIMIT_RETRIES` controls the retry count (default: `2`).
- `database_analyzer.json` is searched locally with lexical ranking, stemming,
  synonyms, and partition-family diversity. No embedding service or vector
  database is required.
- Reports are assembled deterministically and expose database rows as typed
  result records.

## Setup

Install [uv](https://docs.astral.sh/uv/) and synchronize the locked environment:

```bash
uv sync
```

Configure at least:

```dotenv
DATABASE_URL=postgresql://readonly_user:password@host/database
GOOGLE_API_KEY=...
```

Optional model overrides are `EVIDENCE_MODEL`, `INVESTIGATION_MODEL`, and
`ROOT_CAUSE_MODEL`.

## Run

```bash
uv run deep-agent "Why did payment PAY-1042 fail?" \
  --organization-id org-123 --project-id project-456
```

Workflow stages, model activity, tool calls, and quota countdowns are written to
stderr while the final report remains valid JSON on stdout. Pass `--quiet` for
JSON-only automation.

Application code can call `deep_agent.main.investigate_issue` for a final
`RootCauseReport`, or `stream_investigation` for LangGraph node updates.

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
