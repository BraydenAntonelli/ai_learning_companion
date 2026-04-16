from __future__ import annotations

from typing import List


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing space."""
    return " ".join(text.strip().split())


def is_meaningful_text(text: str) -> bool:
    """Return True when text contains non-whitespace content."""
    return bool(normalize_text(text))


def split_tags(tags_text: str) -> List[str]:
    """Split a comma-separated tag string into unique normalized tags."""
    tags: List[str] = []
    seen = set()
    for raw_tag in tags_text.split(","):
        tag = normalize_text(raw_tag).casefold()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Split text into word-aware chunks up to max_chars long."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: List[str] = []
    current_words: List[str] = []
    current_length = 0

    for word in cleaned.split(" "):
        if len(word) > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_length = 0

            start = 0
            while start < len(word):
                chunks.append(word[start : start + max_chars])
                start += max_chars
            continue

        additional_length = len(word) if not current_words else len(word) + 1
        if current_length + additional_length > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def truncate_text(text: str, max_chars: int = 90) -> str:
    """Return a shortened preview of longer text."""
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."
