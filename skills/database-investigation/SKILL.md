---
name: database-investigation
description: Retrieve or verify facts from an analyzed project database.
intents: data_retrieval, informational, analysis, incident_investigation
sources: database
---
# Database investigation

1. Retrieve relevant analyzed schema before querying records.
2. Treat collection, table, field, and relationship metadata as authoritative.
3. Prefer the provider-neutral typed query tool for filters, projection, sorting,
   counting, and distinct values.
4. For MongoDB, never infer BSON types from a field name. Let the typed executor
   coerce values from analyzed metadata. Discover representative values when a
   field has mixed or unknown types.
5. Mark exploratory queries as exploratory. Execute a separate final-answer
   query whose constraints and output exactly match the request.
6. An empty result is a fact only after field names, native types, stored values,
   and requested constraints have been validated.
7. Escalate to another source only when the database cannot establish a required
   business meaning, runtime chronology, or implementation decision.
