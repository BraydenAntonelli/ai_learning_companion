from __future__ import annotations

from typing import List


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing space."""
    return " ".join(text.strip().split())


def is_meaningful_text(text: str) -> bool:
    """Return True when text contains non-whitespace content."""
    return bool(normalize_text(text))


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Split normalized text into fixed-size chunks for future bulk ingestion."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        chunks.append(cleaned[start : start + max_chars])
        start += max_chars
    return chunks
