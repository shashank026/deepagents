from dataclasses import dataclass
from pathlib import Path

from deep_agent.models.query import EvidenceSource, QueryUnderstanding


@dataclass(frozen=True)
class InvestigationSkill:
    name: str
    description: str
    intents: frozenset[str]
    sources: frozenset[str]
    instructions: str


def _metadata(block: str) -> dict[str, str]:
    result = {}
    for line in block.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def discover_skills(root: Path | None = None) -> list[InvestigationSkill]:
    root = root or Path(__file__).resolve().parents[3] / "skills"
    skills = []
    if not root.exists():
        return skills
    for path in sorted(root.glob("*/SKILL.md")):
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---\n") or "\n---\n" not in content[4:]:
            continue
        header, body = content[4:].split("\n---\n", 1)
        meta = _metadata(header)
        split_values = lambda key: frozenset(
            value.strip() for value in meta.get(key, "").split(",") if value.strip()
        )
        skills.append(InvestigationSkill(
            name=meta.get("name", path.parent.name),
            description=meta.get("description", ""),
            intents=split_values("intents"),
            sources=split_values("sources"),
            instructions=body.strip(),
        ))
    return skills


def select_skills(
    understanding: QueryUnderstanding | None,
    sources: set[EvidenceSource],
) -> list[InvestigationSkill]:
    """Progressively disclose skills from typed intent/source metadata."""
    intent = understanding.intent if understanding else "analysis"
    source_names = {item.value for item in sources}
    return [
        skill for skill in discover_skills()
        if (not skill.intents or intent in skill.intents)
        and (not skill.sources or bool(skill.sources & source_names))
    ]
