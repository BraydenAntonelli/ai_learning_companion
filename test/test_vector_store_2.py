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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_store_persists_and_reloads_saved_facts(self) -> None:
        original_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )
        original_store.add([0.0, 0.0, 0.0], "The sky is blue.")

        reloaded_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )

        self.assertEqual(reloaded_store.facts(), ["The sky is blue."])
        self.assertEqual(reloaded_store.search([0.0, 0.0, 0.0], top_k=1)[0][0], "The sky is blue.")

    def test_store_recovers_from_mismatched_metadata_and_index(self) -> None:
        index = faiss.IndexFlatL2(3)
        index.add(np.array([[0.0, 0.0, 0.0]], dtype="float32"))
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(["Fact one", "Fact two"], indent=2),
            encoding="utf-8",
        )

        store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )

        self.assertEqual(store.size(), 0)
        self.assertEqual(store.search([0.0, 0.0, 0.0], top_k=1), [])


if __name__ == "__main__":
    unittest.main()
