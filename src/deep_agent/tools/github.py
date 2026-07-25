import base64
import json
import re
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from deep_agent.services.codebase_context import codebase_source


def _request(source, path: str, query: dict[str, Any] | None = None) -> Any:
    url = f"{source.api_url.rstrip('/')}{path}"
    if query:
        url += f"?{urlencode(query)}"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {source.installation_token}",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "TraceX-DeepAgents",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"GitHub API returned HTTP {exc.code}") from exc


def search_code(query: str, connection_id: str | None = None,
                max_results: int = 30) -> dict[str, Any]:
    """GET /search/code within the connected repository."""
    source = codebase_source(connection_id)
    terms = {term.lower() for term in query.replace("_", " ").split() if len(term) > 1}
    ranked = []
    for item in source.analysis.get("files", []):
        haystack = " ".join([
            str(item.get("filename", "")),
            str(item.get("path", "")),
            str(item.get("language", "")),
        ]).lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda match: (-match[0], match[1].get("path", "")))
    search_queries = [query]
    normalized = re.sub(r"[\"']", " ", query)
    tokens = [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", normalized)
        if token.lower() not in {
            "the", "your", "this", "that", "does", "not", "include",
            "current", "available", "error",
        }
    ]
    if len(tokens) >= 2:
        search_queries.extend([
            " ".join(tokens[:3]),
            " ".join(tokens[-3:]),
        ])
    api_items: list[dict[str, Any]] = []
    attempted_queries: list[str] = []
    seen_paths: set[str] = set()
    for candidate in dict.fromkeys(search_queries):
        attempted_queries.append(candidate)
        result = _request(source, "/search/code", {
            "q": f"{candidate} repo:{source.owner}/{source.repository}",
            "per_page": min(max_results, 100),
        })
        for item in result.get("items", []):
            if item.get("path") not in seen_paths:
                seen_paths.add(item.get("path"))
                api_items.append(item)
        if api_items:
            break
    return {
        "connection_id": source.connection_id,
        "query": query,
        "attempted_queries": attempted_queries,
        "analyzed_matches": [{
            "filename": item.get("filename"),
            "path": item.get("path"),
            "sha": item.get("sha"),
            "language": item.get("language"),
            "score": score,
        } for score, item in ranked[:max_results]],
        "matches": [{
            "filename": item["name"], "path": item["path"], "sha": item["sha"],
            "url": item.get("html_url"),
        } for item in api_items[:max_results]],
    }


def get_commit(ref: str | None = None, connection_id: str | None = None) -> dict[str, Any]:
    """GET /repos/{owner}/{repo}/commits/{ref}."""
    source = codebase_source(connection_id)
    return _request(
        source,
        f"/repos/{quote(source.owner)}/{quote(source.repository)}/commits/"
        f"{quote(ref or source.branch, safe='')}",
    )


def get_contents(path: str, ref: str | None = None,
                 connection_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """GET /repos/{owner}/{repo}/contents/{path}; returns filename and path."""
    source = codebase_source(connection_id)
    suffix = f"/{quote(path)}" if path else ""
    result = _request(
        source,
        f"/repos/{quote(source.owner)}/{quote(source.repository)}/contents{suffix}",
        {"ref": ref or source.branch},
    )
    if isinstance(result, list):
        return [{
            "filename": item["name"], "path": item["path"], "type": item["type"],
            "sha": item["sha"], "size": item.get("size", 0),
        } for item in result]
    content = ""
    if result.get("encoding") == "base64":
        content = base64.b64decode(result.get("content", "")).decode(
            "utf-8", errors="replace"
        )
    return {
        "filename": result["name"], "path": result["path"], "sha": result["sha"],
        "size": result.get("size", 0), "content": content[:200_000],
    }


def get_tree(sha: str, recursive: bool = True,
             connection_id: str | None = None) -> dict[str, Any]:
    """GET /repos/{owner}/{repo}/git/trees/{sha}."""
    source = codebase_source(connection_id)
    result = _request(
        source,
        f"/repos/{quote(source.owner)}/{quote(source.repository)}/git/trees/"
        f"{quote(sha, safe='')}",
        {"recursive": "1"} if recursive else None,
    )
    return {
        "sha": result["sha"], "truncated": result.get("truncated", False),
        "files": [{
            "filename": item["path"].rsplit("/", 1)[-1],
            "path": item["path"], "type": item["type"], "sha": item["sha"],
            "size": item.get("size", 0),
        } for item in result.get("tree", [])],
    }


def get_blob(sha: str, connection_id: str | None = None) -> dict[str, Any]:
    """GET /repos/{owner}/{repo}/git/blobs/{sha}."""
    source = codebase_source(connection_id)
    result = _request(
        source,
        f"/repos/{quote(source.owner)}/{quote(source.repository)}/git/blobs/"
        f"{quote(sha, safe='')}",
    )
    content = base64.b64decode(result.get("content", "")).decode(
        "utf-8", errors="replace"
    ) if result.get("encoding") == "base64" else str(result.get("content", ""))
    return {"sha": result["sha"], "size": result.get("size", 0), "content": content[:200_000]}


def inspect_symbol(
    symbol: str,
    path: str | None = None,
    connection_id: str | None = None,
    context_lines: int = 12,
    max_files: int = 5,
) -> dict[str, Any]:
    """Find a symbol and return focused, line-numbered source snippets."""
    if not symbol.strip():
        raise ValueError("symbol must not be empty")
    source = codebase_source(connection_id)
    if path:
        candidates = [{"path": path}]
    else:
        result = _request(source, "/search/code", {
            "q": f"{symbol} repo:{source.owner}/{source.repository}",
            "per_page": max(1, min(max_files, 10)),
        })
        candidates = result.get("items", [])[:max_files]

    snippets: list[dict[str, Any]] = []
    bounded_context = max(2, min(context_lines, 30))
    pattern = re.compile(re.escape(symbol), re.IGNORECASE)
    for candidate in candidates:
        candidate_path = str(candidate.get("path", ""))
        if not candidate_path:
            continue
        result = get_contents(
            candidate_path,
            ref=source.branch,
            connection_id=source.connection_id,
        )
        if not isinstance(result, dict):
            continue
        lines = str(result.get("content", "")).splitlines()
        matches = [
            index for index, line in enumerate(lines) if pattern.search(line)
        ]
        for index in matches[:5]:
            start = max(0, index - bounded_context)
            end = min(len(lines), index + bounded_context + 1)
            snippets.append({
                "path": candidate_path,
                "sha": result.get("sha"),
                "symbol": symbol,
                "match_line": index + 1,
                "start_line": start + 1,
                "end_line": end,
                "content": "\n".join(
                    f"{line_number + 1}: {lines[line_number]}"
                    for line_number in range(start, end)
                ),
            })
        if len(snippets) >= 12:
            break
    return {
        "connection_id": source.connection_id,
        "owner": source.owner,
        "repository": source.repository,
        "ref": source.branch,
        "symbol": symbol,
        "searched_path": path,
        "snippets": snippets[:12],
        "match_count": len(snippets),
    }
