# Deterministic RCA workflow with LangGraph and DeepAgents

This project investigates database-backed incidents through a parent LangGraph.
The graph—not an agent prompt—enforces the lifecycle:

```text
START → extract business entities → plan investigation → collect evidence
      → validate evidence → investigate → self-check
      → revise and recollect (bounded) OR identify root cause
      → validate root cause → build report → validate report → END
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
- Investigation and root-cause stages use native structured output when
  supported, with JSON extraction, Pydantic validation, and bounded repair as
  the provider-independent fallback.
- Model-produced evidence and hypothesis IDs are checked against workflow state.
- Evidence collection loops are capped at two attempts. Failed assumptions,
  tool errors, and retry counts are typed LangGraph state.
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
`ROOT_CAUSE_MODEL`. `MODEL_STRUCTURED_OUTPUT_MODE=json` forces the portable
JSON/Pydantic path; `MAX_OUTPUT_REPAIRS` defaults to `2`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for diagnosis, migration, folder layout,
and a sample evidence-backed report.

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
## GitHub codebase evidence

When `backend_base` supplies a synced GitHub source, investigations can use
read-only tools for:

- `GET /search/code`
- `GET /repos/{owner}/{repo}/commits/{ref}`
- `GET /repos/{owner}/{repo}/contents/{path}`
- `GET /repos/{owner}/{repo}/git/trees/{sha}`
- `GET /repos/{owner}/{repo}/git/blobs/{sha}`

Contents and tree results preserve both `filename` and repository-relative
`path`. DeepAgents receives only a short-lived installation token for the
duration of the investigation; diagnostic serialization redacts token fields.
## Controlled public-web research

DeepAgents can optionally consult allowlisted official documentation as
supporting context after internal codebase, database, and log evidence. It
never treats web content as proof of a customer-specific root cause.

Configure:

```env
WEB_RESEARCH_ENABLED=true
TAVILY_API_KEY=your_server_side_key
WEB_ALLOWED_DOMAINS=docs.github.com,docs.python.org,www.mongodb.com,docs.langchain.com
```

Search queries containing email addresses, UUIDs, MongoDB ObjectIds,
connection URLs, or credential markers are rejected before a request is made.
Fetched pages must use HTTPS, resolve to public IP addresses, and match the
domain allowlist. Requests have strict redirect, timeout, and response-size
limits. Keep the Tavily key server-side; it is not accepted from investigation
requests or returned to the frontend.
