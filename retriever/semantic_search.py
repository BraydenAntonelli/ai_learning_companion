from __future__ import annotations

from typing import List

from memory.embedder import embed_text
from memory.models import RetrievedMemory
from memory.vector_store import VectorStore
from utils.text_utils import normalize_text


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

    query_vec = embed_text(cleaned_query)
    return store.search(query_vec, top_k=top_k)
