import base64
import inspect
from unittest.mock import patch

from deep_agent.services.codebase_context import (
    bind_codebase_sources, reset_codebase_sources,
)
from deep_agent.tools.github import (
    get_contents, get_tree, inspect_symbol, search_code,
)
from deep_agent.tools.evidence_tools import (
    get_codebase_blob, get_codebase_commit, get_codebase_file,
    get_codebase_tree, inspect_codebase_symbol, search_codebase,
)


class FakeResponse:
    def __init__(self, payload):
        import json
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def source():
    return {
        "connection_id": "code-1", "provider": "github",
        "installation_token": "secret", "api_url": "https://api.github.test",
        "owner": "acme", "repository": "private-api", "branch": "main",
        "analysis": {},
    }


def test_contents_tool_preserves_filename_and_path():
    token = bind_codebase_sources([source()])
    try:
        payload = {
            "name": "users.py", "path": "src/api/users.py", "sha": "blob-1",
            "size": 12, "encoding": "base64",
            "content": base64.b64encode(b"def users(): pass").decode(),
        }
        with patch("deep_agent.tools.github.urlopen", return_value=FakeResponse(payload)):
            result = get_contents("src/api/users.py")
        assert result["filename"] == "users.py"
        assert result["path"] == "src/api/users.py"
        assert "def users" in result["content"]
    finally:
        reset_codebase_sources(token)


def test_tree_tool_adds_filename_to_github_paths():
    token = bind_codebase_sources([source()])
    try:
        payload = {
            "sha": "tree-1", "truncated": False,
            "tree": [{"path": "src/main.py", "type": "blob", "sha": "b1", "size": 8}],
        }
        with patch("deep_agent.tools.github.urlopen", return_value=FakeResponse(payload)):
            result = get_tree("tree-1")
        assert result["files"][0]["filename"] == "main.py"
        assert result["files"][0]["path"] == "src/main.py"
    finally:
        reset_codebase_sources(token)


def test_all_registered_github_evidence_tools_have_descriptions():
    tools = (
        search_codebase,
        get_codebase_file,
        get_codebase_commit,
        get_codebase_tree,
        get_codebase_blob,
        inspect_codebase_symbol,
    )
    assert all(inspect.getdoc(tool) for tool in tools)


def test_code_search_falls_back_from_exact_message_to_meaningful_terms():
    token = bind_codebase_sources([source()])
    try:
        responses = [
            FakeResponse({"items": []}),
            FakeResponse({"items": [{
                "name": "campaign.service.ts",
                "path": "src/campaign/campaign.service.ts",
                "sha": "blob-1",
                "html_url": "https://github.test/file",
            }]}),
        ]
        with patch(
            "deep_agent.tools.github.urlopen",
            side_effect=responses,
        ):
            result = search_code(
                '"Your current plan does not include this campaign type"'
            )
        assert len(result["attempted_queries"]) == 2
        assert result["matches"][0]["path"] == "src/campaign/campaign.service.ts"
    finally:
        reset_codebase_sources(token)


def test_symbol_inspection_returns_focused_line_numbered_evidence():
    token = bind_codebase_sources([source()])
    try:
        content = "\n".join([
            "class CampaignService {",
            "  async create(projectId: string) {",
            "    return this.subscriptionService.validateSubscriptionFeature(",
            "      projectId,",
            "      SUBSCRIPTION_FEATURES_ENUM.apiCampaign,",
            "    );",
            "  }",
            "}",
        ])
        payload = {
            "name": "campaign.service.ts",
            "path": "src/campaign/campaign.service.ts",
            "sha": "blob-1",
            "size": len(content),
            "encoding": "base64",
            "content": base64.b64encode(content.encode()).decode(),
        }
        with patch(
            "deep_agent.tools.github.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = inspect_symbol(
                "validateSubscriptionFeature",
                path="src/campaign/campaign.service.ts",
                context_lines=2,
            )
        assert result["match_count"] == 1
        snippet = result["snippets"][0]
        assert snippet["match_line"] == 3
        assert "3:     return this.subscriptionService" in snippet["content"]
        assert snippet["path"] == "src/campaign/campaign.service.ts"
    finally:
        reset_codebase_sources(token)
