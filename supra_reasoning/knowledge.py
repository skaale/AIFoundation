from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge_tree.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


@dataclass
class KnowledgeEntry:
    id: str
    question: str
    answer: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            question=str(data.get("question", "")).strip(),
            answer=str(data.get("answer", "")).strip(),
            tags=[str(tag) for tag in data.get("tags", [])],
            created_at=str(data.get("created_at") or _now_iso()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass
class KnowledgeNode:
    id: str
    title: str
    summary: str
    children: dict[str, KnowledgeNode] = field(default_factory=dict)
    entries: list[KnowledgeEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeNode:
        children = {
            key: KnowledgeNode.from_dict(child)
            for key, child in (data.get("children") or {}).items()
        }
        entries = [KnowledgeEntry.from_dict(item) for item in data.get("entries") or []]
        return cls(
            id=str(data.get("id", "node")),
            title=str(data.get("title", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            children=children,
            entries=entries,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "children": {key: child.to_dict() for key, child in self.children.items()},
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass
class RetrievalHit:
    topic_path: list[str]
    topic_ids: list[str]
    entry: KnowledgeEntry
    score: float


class KnowledgeTree:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_KNOWLEDGE_PATH)
        self.version = 1
        self.root = KnowledgeNode(id="root", title="ASI Foundation Knowledge", summary="")
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.root = KnowledgeNode(
                id="root",
                title="ASI Foundation Knowledge",
                summary="Sacred and technical wisdom on intelligence, technology, and the divine future.",
            )
            return

        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        self.version = int(payload.get("version", 1))
        self.root = KnowledgeNode.from_dict(payload.get("root") or {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.version, "root": self.root.to_dict()}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def _resolve_node(self, topic_path: str | list[str]) -> KnowledgeNode:
        if isinstance(topic_path, str):
            parts = [part for part in topic_path.split("/") if part]
        else:
            parts = list(topic_path)

        node = self.root
        for part in parts:
            if part not in node.children:
                raise KeyError(f"Unknown knowledge topic '{part}' in path '{'/'.join(parts)}'")
            node = node.children[part]
        return node

    def add_entry(
        self,
        topic_path: str,
        question: str,
        answer: str,
        *,
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            raise ValueError("Both question and answer are required.")

        node = self._resolve_node(topic_path)
        entry = KnowledgeEntry(
            id=uuid.uuid4().hex[:12],
            question=question,
            answer=answer,
            tags=tags or [],
            created_at=_now_iso(),
        )
        node.entries.append(entry)
        self.save()
        return entry

    def list_topic_paths(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []

        def walk(node: KnowledgeNode, ids: list[str], titles: list[str]) -> None:
            if ids:
                choices.append((" > ".join(titles), "/".join(ids)))
            for child_id, child in node.children.items():
                walk(child, ids + [child_id], titles + [child.title])

        walk(self.root, [], [])
        return choices

    def topic_titles(self, topic_path: str) -> list[str]:
        parts = [part for part in topic_path.split("/") if part]
        node = self.root
        titles: list[str] = []
        for part in parts:
            node = node.children[part]
            titles.append(node.title)
        return titles

    def entry_count(self) -> int:
        total = 0

        def walk(node: KnowledgeNode) -> None:
            nonlocal total
            total += len(node.entries)
            for child in node.children.values():
                walk(child)

        walk(self.root)
        return total

    def _topic_score(self, query_tokens: set[str], node: KnowledgeNode) -> float:
        topic_text = " ".join([node.title, node.summary, node.id])
        topic_tokens = _tokenize(topic_text)
        if not query_tokens or not topic_tokens:
            return 0.0
        overlap = len(query_tokens & topic_tokens)
        return overlap / max(1, len(query_tokens))

    def _entry_score(self, query_tokens: set[str], entry: KnowledgeEntry) -> float:
        if not query_tokens:
            return 0.0
        question_tokens = _tokenize(entry.question)
        answer_tokens = _tokenize(entry.answer)
        tag_tokens = _tokenize(" ".join(entry.tags))
        question_overlap = len(query_tokens & question_tokens) / max(1, len(query_tokens))
        answer_overlap = len(query_tokens & answer_tokens) / max(1, len(query_tokens))
        tag_overlap = len(query_tokens & tag_tokens) / max(1, len(query_tokens))
        phrase_bonus = 0.0
        lowered = " ".join(sorted(query_tokens))
        for phrase in (
            " ".join(sorted(question_tokens)),
            " ".join(sorted(answer_tokens)),
        ):
            if lowered and lowered in phrase:
                phrase_bonus = 0.35
        return question_overlap * 1.0 + answer_overlap * 0.45 + tag_overlap * 0.25 + phrase_bonus

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievalHit]:
        query = (query or "").strip()
        if not query:
            return []

        query_tokens = _tokenize(query)
        hits: list[RetrievalHit] = []

        def walk(
            node: KnowledgeNode,
            ids: list[str],
            titles: list[str],
            topic_boost: float,
        ) -> None:
            local_boost = topic_boost + self._topic_score(query_tokens, node) * 0.5
            for entry in node.entries:
                score = self._entry_score(query_tokens, entry) + local_boost
                if score <= 0:
                    continue
                hits.append(
                    RetrievalHit(
                        topic_path=titles.copy(),
                        topic_ids=ids.copy(),
                        entry=entry,
                        score=score,
                    )
                )
            for child_id, child in node.children.items():
                walk(child, ids + [child_id], titles + [child.title], local_boost)

        walk(self.root, [], [], 0.0)
        hits.sort(key=lambda item: item.score, reverse=True)

        seen: set[str] = set()
        unique: list[RetrievalHit] = []
        for hit in hits:
            if hit.entry.id in seen:
                continue
            seen.add(hit.entry.id)
            unique.append(hit)
            if len(unique) >= top_k:
                break
        return unique
