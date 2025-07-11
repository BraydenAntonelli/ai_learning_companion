from typing import List, Tuple
from memory.embedder import embed_text
from memory.vector_store import VectorStore

def search_memory(query: str, store: VectorStore, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Embed the query and return the top_k closest text matches from memory.

    Args:
        query (str): The user's question or input.
        store (VectorStore): An instance of the current vector store.
        top_k (int): Number of closest results to return.

    Returns:
        List of (text, distance) tuples.
    """
    if not query.strip():
        return []

    if len(store.metadata) == 0:
        return []

    query_vec = embed_text(query)
    results = store.search(query_vec, top_k=1)
    return results
