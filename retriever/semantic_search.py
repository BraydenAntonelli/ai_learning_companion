from __future__ import annotations

"""Query cleanup and semantic search helpers.

The main trick here is building a few useful query variants so wording
changes still have a good shot at matching stored memory.
"""

import re
from typing import TYPE_CHECKING, List, Sequence

from memory.models import RetrievedMemory
from utils.text_utils import normalize_text

if TYPE_CHECKING:
    from memory.vector_store import VectorStore

QUESTION_OPENERS = (
    "what is ",
    "what's ",
    "what was ",
    "who is ",
    "who was ",
    "where is ",
    "where was ",
    "when is ",
    "when was ",
    "tell me ",
    "can you tell me ",
    "do you know ",
)
TRAILING_FILLER_WORDS = {"again", "please", "now"}


def _clean_query_fragment(text: str) -> str:
    stripped = re.sub(r"[?.!,:;]+$", "", text.strip())
    return normalize_text(stripped)


def _trim_trailing_fillers(text: str) -> str:
    words = text.split()
    while words and words[-1].casefold().strip("?.!,:;") in TRAILING_FILLER_WORDS:
        words.pop()
    return _clean_query_fragment(" ".join(words))


def _build_query_variants(query: str) -> List[str]:
    cleaned_query = normalize_text(query)
    if not cleaned_query:
        return []

    variants: List[str] = []
    seen = set()

    def add_variant(value: str) -> None:
        cleaned_value = _clean_query_fragment(value)
        if not cleaned_value:
            return
        key = cleaned_value.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(cleaned_value)

    add_variant(cleaned_query)
    add_variant(_trim_trailing_fillers(cleaned_query))

    # Strip common question openers so "what is my favorite food" can still match "my favorite food is pizza".
    for variant in list(variants):
        lowered_variant = variant.casefold()
        for opener in QUESTION_OPENERS:
            if lowered_variant.startswith(opener):
                remainder = variant[len(opener) :]
                add_variant(remainder)
                break

    base_variants = list(variants)
    for variant in base_variants:
        lowered_variant = variant.casefold()
        if lowered_variant.startswith("the "):
            add_variant(variant[4:])
        if lowered_variant.startswith("my favorite "):
            subject = variant[len("my favorite ") :]
            add_variant(f"favorite {subject}")
            add_variant(f"my favorite {subject}")
        if lowered_variant.startswith("the capital of "):
            place = variant[len("the capital of ") :]
            add_variant(f"capital of {place}")

    return variants


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    from memory.embedder import embed_texts as _embed_texts

    return _embed_texts(texts)


def search_memory(
    query: str,
    store: VectorStore,
    top_k: int = 1,
) -> List[RetrievedMemory]:
    """Embed a query and return the closest memory matches."""
    cleaned_query = normalize_text(query)
    if not cleaned_query:
        return []

    if store.size() == 0:
        return []

    aggregated_results: dict[str, RetrievedMemory] = {}
    per_query_top_k = max(top_k, 3)

    query_variants = _build_query_variants(cleaned_query)
    query_vectors = embed_texts(query_variants)

    # We keep the best score per record across all query variants, then sort once at the end.
    for variant, query_vec in zip(query_variants, query_vectors):
        for match in store.search(query_vec, top_k=per_query_top_k):
            existing_match = aggregated_results.get(match.record.id)
            if existing_match is None or match.score > existing_match.score:
                aggregated_results[match.record.id] = match

    ranked_results = sorted(
        aggregated_results.values(),
        key=lambda match: match.score,
        reverse=True,
    )
    return ranked_results[:top_k]
