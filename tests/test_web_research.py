import socket

import pytest

from deep_agent.tools import web_research


def test_web_search_rejects_customer_identifier_before_network(monkeypatch):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "not-used")
    with pytest.raises(ValueError, match="customer identifiers"):
        web_research.search_public_web(
            "mongodb error for customer@example.com"
        )


def test_fetch_rejects_non_allowlisted_domain(monkeypatch):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("WEB_ALLOWED_DOMAINS", "docs.python.org")
    with pytest.raises(ValueError, match="not allowlisted"):
        web_research.fetch_public_page("https://example.com/private")


def test_fetch_rejects_allowlisted_host_resolving_private_ip(monkeypatch):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("WEB_ALLOWED_DOMAINS", "docs.example.com")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(ValueError, match="Private, local"):
        web_research.fetch_public_page("https://docs.example.com/")


def test_web_research_flag_reads_environment(monkeypatch):
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "true")
    assert web_research.web_research_enabled() is True
    monkeypatch.setenv("WEB_RESEARCH_ENABLED", "false")
    assert web_research.web_research_enabled() is False
