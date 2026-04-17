import tempfile
import unittest
from pathlib import Path

"""Persistence-focused checks for the SQLite + FAISS store."""

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.vector_store import VectorStore


class VectorStorePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.index_path = temp_path / "memory.faiss"
        self.metadata_path = temp_path / "memory.sqlite3"
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

    def test_store_rebuilds_from_sqlite_with_batch_embedder_only_when_index_is_missing(self) -> None:
        original_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )
        original_store.add_text("Fact one")
        original_store.add_text("Fact two")

        self.index_path.unlink()

        store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=None,
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )

        self.assertEqual(store.size(), 2)
        self.assertEqual(store.search([0.5, 0.5, 0.0], top_k=1)[0].record.text, "Fact one")

    def test_cleared_sqlite_store_stays_empty_after_reload(self) -> None:
        store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )
        store.add_text("Fact one")
        self.assertEqual(store.size(), 1)

        store.clear()
        self.assertEqual(store.size(), 0)

        reloaded_store = VectorStore(
            dim=3,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )

        self.assertEqual(reloaded_store.size(), 0)


if __name__ == "__main__":
    unittest.main()
