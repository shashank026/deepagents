"""Compatibility export for callers that previously imported ``agent``."""

from deep_agent.workflow.investigation_graph import create_investigation_graph

agent = create_investigation_graph()
