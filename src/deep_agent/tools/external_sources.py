import os
from pathlib import Path
from typing import Any


CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".java", ".go", ".rb", ".sql", ".yaml", ".yml", ".json"}
LOG_SUFFIXES = {".log", ".txt", ".json", ".jsonl"}
SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}


class SourceUnavailableError(RuntimeError):
    pass


def log_source_status() -> dict[str, Any]:
    """Return the actual local-log capability; relevance never implies access."""
    configured = os.getenv("LOG_ROOT", "").strip()
    if not configured:
        return {
            "available": False,
            "provider": None,
            "reason": "No logs source is connected and LOG_ROOT is not configured.",
        }
    root = Path(configured).resolve()
    if not root.exists() or not root.is_dir():
        return {
            "available": False,
            "provider": "local_filesystem",
            "reason": "The configured LOG_ROOT directory is unavailable.",
        }
    return {
        "available": True,
        "provider": "local_filesystem",
        "root": str(root),
        "reason": None,
    }


def _search_files(root: Path, query: str, suffixes: set[str], max_results: int) -> dict[str, Any]:
    if not root.exists() or not root.is_dir():
        return {"query": query, "root": str(root), "matches": [], "unavailable": True,
                "error": f"Search root does not exist: {root}"}
    terms = [term.lower() for term in query.split() if len(term) >= 2]
    matches: list[dict[str, Any]] = []
    files_searched = 0
    for path in root.rglob("*"):
        if len(matches) >= max_results:
            break
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue
        files_searched += 1
        try:
            for line_number, line in enumerate(path.open(errors="ignore"), start=1):
                lowered = line.lower()
                if terms and any(term in lowered for term in terms):
                    matches.append({
                        "path": str(path.relative_to(root)), "line": line_number,
                        "text": line.strip()[:500],
                    })
                    if len(matches) >= max_results:
                        break
        except (OSError, UnicodeError):
            continue
    return {"query": query, "root": str(root), "matches": matches,
            "files_searched": files_searched, "truncated": len(matches) >= max_results,
            "unavailable": False, "error": None}


def search_codebase_files(query: str, max_results: int = 30) -> dict[str, Any]:
    root = Path(os.getenv("CODEBASE_ROOT", Path.cwd())).resolve()
    return _search_files(root, query, CODE_SUFFIXES, min(max(1, max_results), 100))


def search_log_files(query: str, max_results: int = 50) -> dict[str, Any]:
    status = log_source_status()
    if not status["available"]:
        return {"query": query, "matches": [], "unavailable": True,
                "error": status["reason"]}
    root = Path(status["root"])
    return _search_files(root, query, LOG_SUFFIXES, min(max(1, max_results), 100))
