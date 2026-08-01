---
name: runtime-investigation
description: Establish runtime chronology from logs and traces without guessing causes.
intents: incident_investigation, analysis
sources: logs, traces
---
# Runtime investigation

1. Search with schema-verified entity identifiers, bounded time windows, and
   reported error signatures.
2. Preserve timestamps, environment, service/component identity, correlation
   identifiers, and source location.
3. Separate observed events from inferred causal ordering.
4. Correlate events with database state or deployed code before claiming a root
   cause. Absence of a log entry is not proof that an event did not occur when
   retention, sampling, ingestion, or source availability is uncertain.
5. Escalate only for the specific missing decision logic, configuration, or
   deployment change needed to resolve a remaining hypothesis.
