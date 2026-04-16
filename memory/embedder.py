from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List

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
    """Convert text into a local semantic embedding vector."""
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("Cannot embed empty text")

    embedding = get_model().encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embedding.astype("float32").tolist()
