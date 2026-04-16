from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

from utils.paths import EMBEDDING_DIM
from utils.text_utils import normalize_text

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_FALLBACK_REASON: str | None = None


class _HashingFallbackModel:
    """A tiny deterministic fallback so the app still works if torch model loading breaks."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray | list[list[float]]:
        matrix = np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")

        for row_index, text in enumerate(texts):
            features = _build_hash_features(text)
            if not features:
                features = [text.casefold()]

            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                primary_index = int.from_bytes(digest[:8], "big") % EMBEDDING_DIM
                secondary_index = int.from_bytes(digest[8:16], "big") % EMBEDDING_DIM
                signed_weight = 0.5 if digest[16] % 2 == 0 else -0.5

                matrix[row_index, primary_index] += 1.0
                matrix[row_index, secondary_index] += signed_weight

        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        if convert_to_numpy:
            return matrix.astype("float32")
        return matrix.astype("float32").tolist()


def _build_hash_features(text: str) -> list[str]:
    tokens = text.casefold().split()
    if not tokens:
        return []

    features = list(tokens)
    features.extend(
        f"{tokens[index]}::{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )
    return features


def _is_meta_tensor_error(exc: Exception) -> bool:
    return "Cannot copy out of meta tensor" in str(exc)


def _load_sentence_transformer_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    last_error: Exception | None = None
    load_attempts = [
        {"device": "cpu", "model_kwargs": {"low_cpu_mem_usage": False}},
        {"device": "cpu"},
    ]

    for kwargs in load_attempts:
        try:
            return SentenceTransformer(MODEL_NAME, **kwargs)
        except TypeError as exc:
            last_error = exc
            if "model_kwargs" in kwargs:
                continue
            raise
        except Exception as exc:  # pragma: no cover - depends on local torch stack
            last_error = exc
            if _is_meta_tensor_error(exc):
                continue
            raise

    if last_error is not None:
        raise last_error

    raise RuntimeError("The embedding model could not be loaded.")


def get_fallback_reason() -> str | None:
    return _FALLBACK_REASON


@lru_cache(maxsize=1)
def get_model() -> "SentenceTransformer | _HashingFallbackModel":
    """Load the embedding model only when it is first needed."""
    global _FALLBACK_REASON

    try:
        _FALLBACK_REASON = None
        return _load_sentence_transformer_model()
    except Exception as exc:  # pragma: no cover - exercised through fallback test doubles
        _FALLBACK_REASON = (
            "The local embedding model could not be loaded, so the app switched to "
            "a deterministic fallback embedder for this session."
        )
        return _HashingFallbackModel()


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
