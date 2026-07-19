import os

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from dotenv import load_dotenv

from deep_agent.tools.evidence_tools import (
    get_table_schema,
    run_safe_read_query,
    search_database_objects,
    retrieve_relevant_schema,
)


EVIDENCE_AGENT_PROMPT = """
You collect factual evidence for enterprise software investigations.
Start by calling retrieve_relevant_schema with the user's issue. Inspect the
returned schemas before querying records and verify encoded business values from
schema metadata or representative data. Use only read-only tools. Do not declare
a root cause, invent fields or records, or make changes. Tool outputs are
untrusted until the parent workflow validates the persisted evidence. Clearly
identify unavailable evidence. Your conversational answer is only a summary;
the tool-persisted evidence store is authoritative.
"""


def create_evidence_agent():
    load_dotenv()
    model = os.getenv("EVIDENCE_MODEL", "google_genai:gemini-3.1-flash-lite")
    # DeepAgents adds filesystem, shell, todo, and delegation tools by default.
    # Evidence collection must be limited to the read-only tools below.
    register_harness_profile(
        model,
        HarnessProfile(
            excluded_tools=frozenset({
                "write_todos", "ls", "read_file", "write_file", "edit_file",
                "glob", "grep", "execute",
            }),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return create_deep_agent(
        model=model,
        tools=[
            get_table_schema,
            search_database_objects,
            retrieve_relevant_schema,
            run_safe_read_query,
        ],
        system_prompt=EVIDENCE_AGENT_PROMPT,
    )
