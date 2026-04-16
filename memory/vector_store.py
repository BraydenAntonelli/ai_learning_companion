from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import faiss
import numpy as np

from memory.classifier import MemoryClassification, classify_memory_text
from memory.models import MemoryDraft, MemoryRecord, RetrievedMemory, utc_now_iso
from utils.text_utils import normalize_text

EmbeddingFn = Callable[[str], Sequence[float]]
EmbeddingBatchFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class VectorStore:
    """FAISS-backed vector store using cosine similarity over normalized embeddings."""

    def __init__(
        self,
        dim: int,
        index_path: Union[str, PathLike[str]],
        metadata_path: Union[str, PathLike[str]],
        embed_fn: Optional[EmbeddingFn] = None,
        embed_batch_fn: Optional[EmbeddingBatchFn] = None,
    ):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embed_fn = embed_fn
        self.embed_batch_fn = embed_batch_fn

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.index = self._new_index()
        self._records: List[MemoryRecord] = []
        self._load()

    def _new_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self.dim)

    def _reset_in_memory(self) -> None:
        self.index = self._new_index()
        self._records = []

    def _coerce_vector(self, embedding: Sequence[float]) -> np.ndarray:
        vec = np.array(embedding, dtype="float32").reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dim}, got {vec.shape[1]}."
            )

        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Embedding vector norm cannot be zero.")
        return vec / norm

    def _find_record_index_by_id(self, record_id: str) -> Optional[int]:
        for index, record in enumerate(self._records):
            if record.id == record_id:
                return index
        return None

    def _find_record_index_by_text(self, text: str) -> Optional[int]:
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            return None

        target = cleaned_text.casefold()
        for index, record in enumerate(self._records):
            if record.text.casefold() == target:
                return index
        return None

    def _classification_for(self, text: str, source: str) -> MemoryClassification:
        return classify_memory_text(text, source=source)

    def _prepare_record(self, draft: MemoryDraft, existing_record: Optional[MemoryRecord] = None) -> MemoryRecord:
        cleaned_text = normalize_text(draft.text)
        if not cleaned_text:
            raise ValueError("Cannot store empty text")

        classification = self._classification_for(cleaned_text, draft.source)
        category = draft.category or classification.category
        tags = list(classification.tags) + list(draft.tags)

        if existing_record is None:
            return MemoryRecord.create(
                text=cleaned_text,
                category=category,
                source=draft.source,
                tags=tags,
            )

        existing_record.text = cleaned_text
        existing_record.category = category
        existing_record.source = draft.source
        existing_record.tags = MemoryRecord.from_dict(
            {
                "id": existing_record.id,
                "text": existing_record.text,
                "category": existing_record.category,
                "source": existing_record.source,
                "tags": tags,
                "created_at": existing_record.created_at,
                "updated_at": existing_record.updated_at,
            }
        ).tags
        existing_record.updated_at = utc_now_iso()
        return existing_record

    def _embed_many(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not texts:
            return []

        if self.embed_batch_fn is not None:
            embeddings = self.embed_batch_fn(texts)
            return [self._coerce_vector(embedding) for embedding in embeddings]

        if self.embed_fn is None:
            raise ValueError("This operation requires an embedding function.")

        return [self._coerce_vector(self.embed_fn(text)) for text in texts]

    def _rebuild_index_from_records(self, records: List[MemoryRecord]) -> None:
        new_index = self._new_index()
        if records:
            matrix = np.vstack(self._embed_many([record.text for record in records]))
            new_index.add(matrix)

        self.index = new_index
        self._records = list(records)
        self._save()

    def _load(self) -> None:
        self._reset_in_memory()

        metadata_exists = self.metadata_path.exists()
        index_exists = self.index_path.exists()

        if not metadata_exists and not index_exists:
            return

        records = self._load_metadata()
        if records is None:
            self._save()
            return

        if not index_exists:
            if records and self.embed_fn is not None:
                self._rebuild_index_from_records(records)
            else:
                self._records = records
                self._save()
            return

        index = self._load_index()
        metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT) if index else None
        needs_rebuild = (
            index is None
            or index.d != self.dim
            or index.ntotal != len(records)
            or metric_type != faiss.METRIC_INNER_PRODUCT
        )

        if needs_rebuild:
            if records and self.embed_fn is not None:
                self._rebuild_index_from_records(records)
            else:
                self._records = records
                self._save()
            return

        self.index = index
        self._records = records

    def _load_metadata(self) -> Optional[List[MemoryRecord]]:
        try:
            with self.metadata_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(raw, list):
            return None

        records: List[MemoryRecord] = []
        for item in raw:
            try:
                if isinstance(item, str):
                    classification = classify_memory_text(item, source="legacy")
                    records.append(
                        MemoryRecord.create(
                            text=item,
                            category=classification.category,
                            source="legacy",
                            tags=classification.tags,
                        )
                    )
                elif isinstance(item, dict):
                    records.append(MemoryRecord.from_dict(item))
                else:
                    return None
            except ValueError:
                return None

        return records

    def _load_index(self) -> Optional[faiss.Index]:
        try:
            return faiss.read_index(str(self.index_path))
        except (OSError, RuntimeError):
            return None

    def add(
        self,
        embedding: Sequence[float],
        text: str,
        *,
        source: str = "manual",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        if self._find_record_index_by_text(text) is not None:
            return False

        record = self._prepare_record(
            MemoryDraft(
                text=text,
                source=source,
                category=category,
                tags=list(tags or []),
            )
        )

        self.index.add(self._coerce_vector(embedding))
        self._records.append(record)
        self._save()
        return True

    def add_text(
        self,
        text: str,
        *,
        source: str = "manual",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> tuple[bool, MemoryRecord]:
        if self.embed_fn is None:
            raise ValueError("This operation requires an embedding function.")

        existing_index = self._find_record_index_by_text(text)
        if existing_index is not None:
            return False, self._records[existing_index]

        record = self._prepare_record(
            MemoryDraft(
                text=text,
                source=source,
                category=category,
                tags=list(tags or []),
            )
        )
        self.index.add(self._coerce_vector(self.embed_fn(record.text)))
        self._records.append(record)
        self._save()
        return True, record

    def add_many(self, drafts: Sequence[MemoryDraft]) -> tuple[int, int]:
        if not drafts:
            return 0, 0

        seen = {record.text.casefold() for record in self._records}
        new_records: List[MemoryRecord] = []
        duplicates = 0

        for draft in drafts:
            cleaned_text = normalize_text(draft.text)
            if not cleaned_text:
                continue

            key = cleaned_text.casefold()
            if key in seen:
                duplicates += 1
                continue

            seen.add(key)
            new_records.append(
                self._prepare_record(
                    MemoryDraft(
                        text=cleaned_text,
                        source=draft.source,
                        category=draft.category,
                        tags=list(draft.tags),
                    )
                )
            )

        if not new_records:
            return 0, duplicates

        matrix = np.vstack(self._embed_many([record.text for record in new_records]))
        self.index.add(matrix)
        self._records.extend(new_records)
        self._save()
        return len(new_records), duplicates

    def search(self, embedding: Sequence[float], top_k: int = 1) -> List[RetrievedMemory]:
        if self.index.ntotal == 0 or not self._records:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vec = self._coerce_vector(embedding)

        k = min(top_k, self.index.ntotal, len(self._records))
        scores, indices = self.index.search(vec, k)

        results: List[RetrievedMemory] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self._records):
                results.append(
                    RetrievedMemory(
                        record=self._records[idx],
                        score=float(score),
                    )
                )
        return results

    def delete_record(self, record_id: str) -> bool:
        record_index = self._find_record_index_by_id(record_id)
        if record_index is None:
            return False

        remaining_records = [
            record for index, record in enumerate(self._records) if index != record_index
        ]
        self._rebuild_index_from_records(remaining_records)
        return True

    def update_record(
        self,
        record_id: str,
        new_text: str,
        *,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        record_index = self._find_record_index_by_id(record_id)
        if record_index is None:
            return "missing"

        cleaned_new_text = normalize_text(new_text)
        if not cleaned_new_text:
            raise ValueError("Cannot store empty text")

        duplicate_index = self._find_record_index_by_text(cleaned_new_text)
        if duplicate_index is not None and duplicate_index != record_index:
            return "duplicate"

        existing_record = self._records[record_index]
        updated_record = self._prepare_record(
            MemoryDraft(
                text=cleaned_new_text,
                source=source or existing_record.source,
                category=category or existing_record.category,
                tags=list(tags or existing_record.tags),
            ),
            existing_record=existing_record,
        )

        updated_records = list(self._records)
        updated_records[record_index] = updated_record
        self._rebuild_index_from_records(updated_records)
        return "updated"

    def clear(self) -> None:
        self._reset_in_memory()
        self._save()

    def size(self) -> int:
        return len(self._records)

    def facts(self) -> List[str]:
        return [record.text for record in self._records]

    def records(self) -> List[MemoryRecord]:
        return list(self._records)

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                [record.to_dict() for record in self._records],
                file,
                indent=2,
                ensure_ascii=False,
            )
