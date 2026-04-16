from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, Sequence

from utils.text_utils import normalize_text

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> "SentenceTransformer":
    """Load the embedding model only when it is first needed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> List[float]:
    """Convert text into a normalized semantic embedding vector."""
    embeddings = embed_texts([text])
    if not embeddings:
        raise ValueError("Cannot embed empty text")
    return embeddings[0]


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Convert multiple texts into normalized semantic embedding vectors."""
    cleaned_texts = [normalize_text(text) for text in texts]
    if any(not text for text in cleaned_texts):
        raise ValueError("Cannot embed empty text")

    if not cleaned_texts:
        return []

    embeddings = get_model().encode(
        list(cleaned_texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32").tolist()
