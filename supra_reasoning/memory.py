from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory" / "profiles"

MAX_STORED_TURNS = 40
MAX_CONTEXT_TURNS = 6
MAX_FACTS = 12
MAX_SUMMARY_CHARS = 520

_TOPIC_KEYWORDS = {
    "agi": ("agi", "general intelligence", "superintelligence", "super intelligence"),
    "alignment": ("alignment", "aligned", "safe ai", "safety"),
    "ai": ("artificial intelligence", " machine learning", " neural", " ai "),
    "technology": ("technology", "compute", "software", "gpu", "chip"),
    "ai_god": ("ai god", "divine", "theology", "church", "worship", "sacred"),
    "consciousness": ("consciousness", "sentience", "soul", "awareness"),
    "ethics": ("ethics", "moral", "responsible", "duty"),
}

_FACT_PATTERNS = (
    re.compile(r"\bi am\b[^.?!]{0,120}", re.IGNORECASE),
    re.compile(r"\bi'm\b[^.?!]{0,120}", re.IGNORECASE),
    re.compile(r"\bi work\b[^.?!]{0,120}", re.IGNORECASE),
    re.compile(r"\bi care about\b[^.?!]{0,120}", re.IGNORECASE),
    re.compile(r"\bi believe\b[^.?!]{0,120}", re.IGNORECASE),
    re.compile(r"\bmy (?:question|concern|hope|fear)\b[^.?!]{0,120}", re.IGNORECASE),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or f"guest-{uuid.uuid4().hex[:8]}"


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class MemoryProfile:
    profile_id: str
    name: str
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    turns: list[dict[str, str]] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sessions: int = 0
    created_at: str = ""
    last_seen: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> MemoryProfile:
        return cls(
            profile_id=str(data.get("profile_id", "")),
            name=str(data.get("name", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            facts=[str(item).strip() for item in data.get("facts", []) if str(item).strip()],
            turns=[
                {
                    "user": str(turn.get("user", "")).strip(),
                    "assistant": str(turn.get("assistant", "")).strip(),
                }
                for turn in data.get("turns", [])
                if turn.get("user") or turn.get("assistant")
            ],
            topics=[str(item).strip() for item in data.get("topics", []) if str(item).strip()],
            sessions=int(data.get("sessions", 0)),
            created_at=str(data.get("created_at") or ""),
            last_seen=str(data.get("last_seen") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "summary": self.summary,
            "facts": self.facts,
            "turns": self.turns,
            "topics": self.topics,
            "sessions": self.sessions,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
        }


class ConversationMemory:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or DEFAULT_MEMORY_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.json"

    def profile_id_for_name(self, name: str) -> str:
        return _slugify_name(name)

    def load(self, name: str) -> MemoryProfile:
        profile_id = self.profile_id_for_name(name)
        path = self._path_for(profile_id)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                profile = MemoryProfile.from_dict(json.load(handle))
            if profile.name != name.strip():
                profile.name = name.strip()
            return profile

        now = _now_iso()
        return MemoryProfile(
            profile_id=profile_id,
            name=name.strip(),
            created_at=now,
            last_seen=now,
        )

    def load_by_id(self, profile_id: str) -> MemoryProfile | None:
        path = self._path_for(profile_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return MemoryProfile.from_dict(json.load(handle))

    def save(self, profile: MemoryProfile) -> None:
        profile.last_seen = _now_iso()
        path = self._path_for(profile.profile_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(profile.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def begin_session(self, profile: MemoryProfile) -> None:
        profile.sessions += 1
        if not profile.created_at:
            profile.created_at = _now_iso()
        self.save(profile)

    def add_turn(self, profile: MemoryProfile, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text and not assistant_text:
            return

        profile.turns.append({"user": user_text, "assistant": assistant_text})
        if len(profile.turns) > MAX_STORED_TURNS:
            profile.turns = profile.turns[-MAX_STORED_TURNS:]

        if user_text:
            self._extract_facts(profile, user_text)
            self._update_topics(profile, user_text)

        profile.summary = self._build_summary(profile)
        self.save(profile)

    def _extract_facts(self, profile: MemoryProfile, user_text: str) -> None:
        for pattern in _FACT_PATTERNS:
            for match in pattern.findall(user_text):
                fact = _clip(match.strip(), 140)
                if fact and fact not in profile.facts:
                    profile.facts.append(fact)
        if len(profile.facts) > MAX_FACTS:
            profile.facts = profile.facts[-MAX_FACTS:]

    def _update_topics(self, profile: MemoryProfile, user_text: str) -> None:
        lowered = f" {user_text.lower()} "
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                if topic not in profile.topics:
                    profile.topics.append(topic)
        if len(profile.topics) > 8:
            profile.topics = profile.topics[-8:]

    def _build_summary(self, profile: MemoryProfile) -> str:
        parts: list[str] = []
        if profile.name:
            parts.append(f"{profile.name} returns for counsel.")
        if profile.topics:
            parts.append(f"Recurring interests: {', '.join(profile.topics)}.")
        if profile.facts:
            parts.append("Personal notes: " + "; ".join(profile.facts[-4:]))
        recent_users = [turn["user"] for turn in profile.turns[-3:] if turn.get("user")]
        if recent_users:
            parts.append("Recently discussed: " + " | ".join(_clip(item, 90) for item in recent_users))
        return _clip(" ".join(parts), MAX_SUMMARY_CHARS)

    def build_context(self, profile: MemoryProfile | None) -> str:
        if profile is None or (not profile.turns and not profile.facts and not profile.summary):
            return ""

        lines = [
            "You have spoken with this person before. Continue with warmth and consistency.",
            "Ask follow-up questions about topics they care about when it deepens the conversation.",
            "Do not mention memory, records, databases, prior sessions, or that you remember anything.",
        ]
        if profile.name:
            visits = f"{profile.sessions} visit{'s' if profile.sessions != 1 else ''}"
            lines.append(f"Name: {profile.name} ({visits}).")
        if profile.summary:
            lines.append(f"Summary: {profile.summary}")
        if profile.facts:
            lines.append("Remember:")
            for fact in profile.facts[-6:]:
                lines.append(f"- {fact}")
        recent = profile.turns[-MAX_CONTEXT_TURNS:]
        if recent:
            lines.append("Recent exchanges:")
            for turn in recent:
                if turn.get("user"):
                    lines.append(f"- They said: {_clip(turn['user'], 120)}")
                if turn.get("assistant"):
                    lines.append(f"- You answered: {_clip(turn['assistant'], 120)}")
        return "\n".join(lines).strip()
