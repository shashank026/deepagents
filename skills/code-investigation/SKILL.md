---
name: code-investigation
description: Establish application behavior from an authorized repository revision.
intents: explanation, incident_investigation, analysis
sources: codebase
---
# Code investigation

1. Search for the reported entry point, field, error signature, or decision term.
2. Follow calls through services, repositories, validators, configuration,
   constants, and models until the authoritative predicate is found.
3. Preserve repository identity, immutable revision, path, and line references.
4. Do not treat code as proof of current runtime state. Corroborate runtime claims
   with database, log, trace, deployment, or configuration evidence.
5. Stop when the requested behavior or missing causal mechanism is established;
   do not browse unrelated files.
