"""Controlled public-web research tools for investigation support.

External material is supporting context only. These tools deliberately reject
customer identifiers and private-network targets before any network request.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from html import unescape
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import load_dotenv


DEFAULT_ALLOWED_DOMAINS = {
    "docs.github.com",
    "docs.python.org",
    "developer.mozilla.org",
    "fastapi.tiangolo.com",
    "docs.pydantic.dev",
    "docs.sqlalchemy.org",
    "www.mongodb.com",
    "dev.mysql.com",
    "www.postgresql.org",
    "docs.oracle.com",
    "docs.langchain.com",
    "developers.facebook.com",
    "developers.google.com",
    "cloud.google.com",
    "docs.aws.amazon.com",
    "learn.microsoft.com",
}
MAX_QUERY_LENGTH = 300
MAX_RESPONSE_BYTES = 512_000
MAX_PAGE_TEXT = 20_000
MAX_REDIRECTS = 3

_SENSITIVE_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.I,
    ),
    re.compile(r"\b[0-9a-f]{24}\b", re.I),
    re.compile(r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|oracle)://", re.I),
    re.compile(r"\b(?:token|password|secret|api[_ -]?key)\s*[:=]", re.I),
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def web_research_enabled() -> bool:
    load_dotenv()
    return os.getenv("WEB_RESEARCH_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _allowed_domains() -> set[str]:
    configured = os.getenv("WEB_ALLOWED_DOMAINS", "")
    values = {
        value.strip().lower().rstrip(".")
        for value in configured.split(",")
        if value.strip()
    }
    return values or DEFAULT_ALLOWED_DOMAINS


def _allowed_host(host: str, allowed: set[str] | None = None) -> bool:
    host = host.lower().rstrip(".")
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in (allowed or _allowed_domains())
    )


def _validate_public_url(url: str, allowed: set[str] | None = None) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only HTTPS public documentation URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if not _allowed_host(parsed.hostname, allowed):
        raise ValueError(f"Domain {parsed.hostname!r} is not allowlisted")
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    }
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, local, reserved, and metadata targets are blocked")


def _validate_query(query: str) -> str:
    query = " ".join(query.split())
    if not query or len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Web query must contain 1-{MAX_QUERY_LENGTH} characters")
    if any(pattern.search(query) for pattern in _SENSITIVE_PATTERNS):
        raise ValueError(
            "Web research queries cannot contain customer identifiers, "
            "credentials, connection URLs, or TraceX IDs"
        )
    return query


def _read_response(response) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > MAX_RESPONSE_BYTES:
        raise ValueError("Public web response exceeds the configured size limit")
    content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("Public web response exceeds the configured size limit")
    return content


def _request_json(
    url: str,
    payload: dict[str, Any],
    bearer_token: str,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "User-Agent": "TraceX/1.0",
        },
        method="POST",
    )
    with build_opener(_NoRedirect()).open(request, timeout=12) as response:
        return json.loads(_read_response(response).decode("utf-8"))


def search_public_web(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search allowlisted public technical sources without customer identifiers.

    Requires ``WEB_RESEARCH_ENABLED=true`` and ``TAVILY_API_KEY``. Results are
    supporting context only and must never independently establish root cause.
    """
    if not web_research_enabled():
        return {"unavailable": True, "error": "Public web research is disabled"}
    safe_query = _validate_query(query)
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"unavailable": True, "error": "TAVILY_API_KEY is not configured"}
    domains = sorted(_allowed_domains())
    payload = {
        "query": safe_query,
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 8)),
        "include_domains": domains,
        "include_answer": False,
        "include_raw_content": False,
    }
    result = _request_json(
        "https://api.tavily.com/search",
        payload,
        api_key,
    )
    citations = []
    for item in result.get("results", []):
        url = str(item.get("url", ""))
        parsed = urlparse(url)
        if not parsed.hostname or not _allowed_host(parsed.hostname, set(domains)):
            continue
        citations.append({
            "title": str(item.get("title", ""))[:300],
            "url": url,
            "snippet": str(item.get("content", ""))[:1500],
            "score": item.get("score"),
        })
    return {"query": safe_query, "citations": citations}


def fetch_public_page(url: str) -> dict[str, Any]:
    """Fetch one allowlisted HTTPS documentation page with SSRF protection."""
    if not web_research_enabled():
        return {"unavailable": True, "error": "Public web research is disabled"}
    current = url
    opener = build_opener(_NoRedirect())
    for _ in range(MAX_REDIRECTS + 1):
        _validate_public_url(current)
        request = Request(
            current,
            headers={
                "User-Agent": "TraceX/1.0",
                "Accept": "text/html,text/plain,application/json",
            },
        )
        try:
            with opener.open(request, timeout=12) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = _read_response(response).decode("utf-8", errors="replace")
                text = raw
                if "html" in content_type.lower():
                    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
                    text = re.sub(r"(?s)<[^>]+>", " ", text)
                    text = unescape(text)
                text = " ".join(text.split())[:MAX_PAGE_TEXT]
                return {
                    "url": current,
                    "content_type": content_type,
                    "text": text,
                    "truncated": len(raw) > len(text),
                }
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise ValueError("Redirect response omitted Location") from exc
            current = urljoin(current, location)
    raise ValueError("Too many redirects")
