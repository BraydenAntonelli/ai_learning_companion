from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np

from utils.text_utils import normalize_text

EmbeddingFn = Callable[[str], Sequence[float]]


class VectorStore:
    """Simple FAISS-backed vector store for semantic memory."""

    def __init__(
        self,
        dim: int,
        index_path: Union[str, PathLike[str]],
        metadata_path: Union[str, PathLike[str]],
    ):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata: List[str] = []
        self._load()

    def _reset_in_memory(self) -> None:
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

    def _coerce_vector(self, embedding: Sequence[float]) -> np.ndarray:
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dim}, got {vec.shape[1]}."
            )
        return vec

    def _find_fact_index(self, text: str) -> Optional[int]:
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            return None

        target = cleaned_text.casefold()
        for index, fact in enumerate(self.metadata):
            if fact.casefold() == target:
                return index
        return None

    def _normalize_unique_facts(self, facts: List[str]) -> List[str]:
        cleaned_facts: List[str] = []
        seen = set()
        for fact in facts:
            cleaned_fact = normalize_text(fact)
            if not cleaned_fact:
                continue

            key = cleaned_fact.casefold()
            if key in seen:
                continue

            seen.add(key)
            cleaned_facts.append(cleaned_fact)
        return cleaned_facts

    def _rebuild_from_facts(self, facts: List[str], embed_text_fn: EmbeddingFn) -> None:
        cleaned_facts = self._normalize_unique_facts(facts)
        new_index = faiss.IndexFlatL2(self.dim)

        if cleaned_facts:
            matrix = np.vstack([self._coerce_vector(embed_text_fn(fact)) for fact in cleaned_facts])
            new_index.add(matrix)

        self.index = new_index
        self.metadata = cleaned_facts
        self._save()

    def _load(self) -> None:
        self._reset_in_memory()

        metadata_exists = self.metadata_path.exists()
        index_exists = self.index_path.exists()

        if not metadata_exists and not index_exists:
            return

        if metadata_exists != index_exists:
            self._save()
            return

        metadata = self._load_metadata()
        index = self._load_index()

        if metadata is None or index is None:
            self._save()
            return

        if index.d != self.dim or index.ntotal != len(metadata):
            self._save()
            return

        self.metadata = metadata
        self.index = index

    def _load_metadata(self) -> Optional[List[str]]:
        try:
            with self.metadata_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(raw, list):
            return None

        cleaned_metadata: List[str] = []
        for item in raw:
            if not isinstance(item, str):
                return None

            cleaned_item = normalize_text(item)
            if not cleaned_item:
                return None

            cleaned_metadata.append(cleaned_item)

        return cleaned_metadata

    def _load_index(self) -> Optional[faiss.Index]:
        try:
            return faiss.read_index(str(self.index_path))
        except (OSError, RuntimeError):
            return None

    def add(self, embedding: Sequence[float], text: str) -> bool:
        """Add a fact to memory. Returns False when the fact already exists."""
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            raise ValueError("Cannot store empty text")

        if self._find_fact_index(cleaned_text) is not None:
            return False

        vec = self._coerce_vector(embedding)
        self.index.add(vec)
        self.metadata.append(cleaned_text)
        self._save()
        return True

    def search(self, embedding: Sequence[float], top_k: int = 1) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0 or len(self.metadata) == 0:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vec = self._coerce_vector(embedding)

        k = min(top_k, self.index.ntotal, len(self.metadata))
        distances, indices = self.index.search(vec, k)

        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        return results

    def delete_fact(self, text: str, embed_text_fn: EmbeddingFn) -> bool:
        fact_index = self._find_fact_index(text)
        if fact_index is None:
            return False

        remaining_facts = [
            fact for index, fact in enumerate(self.metadata) if index != fact_index
        ]
        self._rebuild_from_facts(remaining_facts, embed_text_fn)
        return True

    def update_fact(self, old_text: str, new_text: str, embed_text_fn: EmbeddingFn) -> str:
        fact_index = self._find_fact_index(old_text)
        if fact_index is None:
            return "missing"

        cleaned_new_text = normalize_text(new_text)
        if not cleaned_new_text:
            raise ValueError("Cannot store empty text")

        duplicate_index = self._find_fact_index(cleaned_new_text)
        if duplicate_index is not None and duplicate_index != fact_index:
            return "duplicate"

        updated_facts = list(self.metadata)
        updated_facts[fact_index] = cleaned_new_text
        self._rebuild_from_facts(updated_facts, embed_text_fn)
        return "updated"

    def clear(self) -> None:
        self._reset_in_memory()
        self._save()

    def size(self) -> int:
        return len(self.metadata)

    def facts(self) -> List[str]:
        return list(self.metadata)

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2, ensure_ascii=False)
