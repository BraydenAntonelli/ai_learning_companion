from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from utils.text_utils import normalize_text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tags(tags: List[str]) -> List[str]:
    normalized_tags: List[str] = []
    seen = set()
    for raw_tag in tags:
        tag = normalize_text(raw_tag).casefold()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized_tags.append(tag)
    return normalized_tags


@dataclass
class MemoryDraft:
    text: str
    source: str = "manual"
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryRecord:
    id: str
    text: str
    category: str
    source: str
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(
        cls,
        text: str,
        category: str,
        source: str = "manual",
        tags: Optional[List[str]] = None,
    ) -> "MemoryRecord":
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            raise ValueError("Cannot create a memory record from empty text")

        timestamp = utc_now_iso()
        return cls(
            id=uuid4().hex,
            text=cleaned_text,
            category=category,
            source=source,
            tags=_normalize_tags(list(tags or [])),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        text = normalize_text(str(data.get("text", "")))
        if not text:
            raise ValueError("Memory record text cannot be empty")

        category = normalize_text(str(data.get("category", "factual_statement"))) or "factual_statement"
        source = normalize_text(str(data.get("source", "manual"))) or "manual"
        created_at = str(data.get("created_at") or utc_now_iso())
        updated_at = str(data.get("updated_at") or created_at)

        return cls(
            id=str(data.get("id") or uuid4().hex),
            text=text,
            category=category,
            source=source,
            tags=_normalize_tags(list(data.get("tags", []))),
            created_at=created_at,
            updated_at=updated_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "source": self.source,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RetrievedMemory:
    record: MemoryRecord
    score: float
