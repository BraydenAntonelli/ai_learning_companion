from __future__ import annotations

"""Lightweight rule-based memory classification.

Nothing fancy here on purpose. The rules are simple, local, and easy to tweak.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from utils.text_utils import normalize_text

CATEGORY_OPTIONS = [
    "personal_context",
    "academic_concept",
    "factual_statement",
    "temporary_note",
    "document_excerpt",
    "question_like_input",
]

QUESTION_PREFIXES = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "am",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "tell",
    "explain",
}

TEACH_PREFIXES = [
    "remember that",
    "remember",
    "store this",
    "save this",
    "learn this",
    "note that",
    "fact:",
]


@dataclass
class MemoryClassification:
    category: str
    tags: List[str] = field(default_factory=list)


def _unique_tags(tags: List[str]) -> List[str]:
    unique_tags: List[str] = []
    seen = set()
    for raw_tag in tags:
        tag = normalize_text(raw_tag).casefold()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        unique_tags.append(tag)
    return unique_tags


def looks_like_question(text: str) -> bool:
    # A quick first pass is enough for this app. We just need a solid guess, not deep NLP.
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return True

    first_word = cleaned.split(" ", 1)[0].casefold().strip(",:.-")
    return first_word in QUESTION_PREFIXES


def strip_teach_prefix(text: str) -> str:
    cleaned = normalize_text(text)
    lowered = cleaned.casefold()
    for prefix in sorted(TEACH_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix):
            stripped = cleaned[len(prefix) :].lstrip(" :-,")
            return normalize_text(stripped)
    return cleaned


def detect_message_intent(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return "teach"
    if looks_like_question(cleaned):
        return "ask"
    return "teach"


def classify_memory_text(text: str, source: str = "manual") -> MemoryClassification:
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("Cannot classify empty text")

    lowered = cleaned.casefold()
    tags: List[str] = []
    if source.startswith("upload:"):
        tags.append("document")
        filename_stem = Path(source.removeprefix("upload:")).stem
        if filename_stem:
            tags.append(filename_stem.replace("_", "-").replace(" ", "-").casefold())

    if source.startswith("upload:"):
        # Uploaded text is treated as document memory first, even if the wording looks like a question.
        return MemoryClassification(
            category="document_excerpt",
            tags=_unique_tags(tags),
        )

    if looks_like_question(cleaned):
        tags.append("question-like")
        return MemoryClassification(
            category="question_like_input",
            tags=_unique_tags(tags),
        )

    temporary_markers = (
        "remember to",
        "don't forget",
        "do not forget",
        "todo",
        "note that",
        "reminder",
    )
    if lowered.startswith(temporary_markers):
        tags.append("note")
        return MemoryClassification(
            category="temporary_note",
            tags=_unique_tags(tags),
        )

    personal_context_phrases = (
        "my favorite",
        "i prefer",
        "i like",
        "i love",
        "my name is",
        "i am",
        "i'm",
    )
    preference_markers = (
        "favorite",
        "prefer",
        "like",
        "love",
        "name is",
        "i am",
        "i'm",
    )
    if any(phrase in lowered for phrase in personal_context_phrases) or (
        lowered.startswith(("my ", "i "))
        and any(marker in lowered for marker in preference_markers)
    ):
        tags.extend(["personal", "profile"])
        return MemoryClassification(
            category="personal_context",
            tags=_unique_tags(tags),
        )

    academic_markers = (
        "photosynthesis",
        "equation",
        "theory",
        "algorithm",
        "process",
        "learning",
        "model",
        "embedding",
        "vector",
        "database",
        "semantic",
        "retrieval",
        "mass-energy",
        "concept",
    )
    if any(marker in lowered for marker in academic_markers):
        tags.extend(["study", "concept"])
        return MemoryClassification(
            category="academic_concept",
            tags=_unique_tags(tags),
        )

    return MemoryClassification(
        category="factual_statement",
        tags=_unique_tags(tags),
    )
