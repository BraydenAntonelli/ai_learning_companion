import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.vector_store import VectorStore


class VectorStorePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.index_path = temp_path / "memory.faiss"
        self.metadata_path = temp_path / "memory.json"
        self.embedding_map = {
            "The sky is blue.": [1.0, 0.0, 0.0],
            "Fact one": [0.5, 0.5, 0.0],
            "Fact two": [0.0, 1.0, 0.0],
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_store_persists_and_reloads_structured_records(self) -> None:
        original_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )
        original_store.add_text("The sky is blue.")

        reloaded_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )

        self.assertEqual(reloaded_store.facts(), ["The sky is blue."])
        self.assertEqual(reloaded_store.records()[0].category, "factual_statement")
        self.assertEqual(
            reloaded_store.search([1.0, 0.0, 0.0], top_k=1)[0].record.text,
            "The sky is blue.",
        )

    def test_store_migrates_legacy_string_metadata_and_rebuilds_index(self) -> None:
        legacy_index = faiss.IndexFlatL2(3)
        legacy_index.add(np.array([[0.0, 0.0, 1.0]], dtype="float32"))
        faiss.write_index(legacy_index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(["Fact one", "Fact two"], indent=2),
            encoding="utf-8",
        )

        store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )

        self.assertEqual(store.size(), 2)
        self.assertEqual(store.records()[0].category, "factual_statement")
        self.assertEqual(store.search([0.5, 0.5, 0.0], top_k=1)[0].record.text, "Fact one")


if __name__ == "__main__":
    unittest.main()
