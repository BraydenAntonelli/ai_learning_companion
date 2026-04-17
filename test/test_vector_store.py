import tempfile
import unittest
from pathlib import Path

"""Core vector-store behavior tests for add, update, delete, and search."""

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.models import MemoryDraft
from memory.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.embedding_map = {
            "Rock music is my favorite.": [1.0, 0.0, 0.0],
            "I also enjoy classical music.": [0.0, 1.0, 0.0],
            "My favorite music is metal.": [0.9, 0.1, 0.0],
            "Document chunk one.": [0.0, 0.0, 1.0],
            "Document chunk two.": [0.0, 0.2, 0.8],
        }
        self.store = VectorStore(
            dim=3,
            index_path=temp_path / "memory.faiss",
            metadata_path=temp_path / "memory.sqlite3",
            embed_fn=lambda text: self.embedding_map[text],
            embed_batch_fn=lambda texts: [self.embedding_map[text] for text in texts],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_and_search_return_closest_match(self) -> None:
        added, _ = self.store.add_text("Rock music is my favorite.")
        self.assertTrue(added)
        self.store.add_text("I also enjoy classical music.")

        results = self.store.search([0.99, 0.01, 0.0], top_k=2)

        self.assertEqual(results[0].record.text, "Rock music is my favorite.")
        self.assertGreater(results[0].score, results[1].score)

    def test_duplicate_detection_ignores_case_and_extra_spacing(self) -> None:
        self.assertTrue(self.store.add([1.0, 0.0, 0.0], "Rock music is my favorite."))
        self.assertFalse(self.store.add([1.0, 0.0, 0.0], "  rock music is my favorite.  "))

    def test_add_many_batches_document_chunks(self) -> None:
        added_count, duplicate_count = self.store.add_many(
            [
                MemoryDraft(
                    text="Document chunk one.",
                    source="upload:notes.txt",
                    category="document_excerpt",
                    tags=["document", "notes"],
                ),
                MemoryDraft(
                    text="Document chunk two.",
                    source="upload:notes.txt",
                    category="document_excerpt",
                    tags=["document", "notes"],
                ),
                MemoryDraft(
                    text="Document chunk one.",
                    source="upload:notes.txt",
                    category="document_excerpt",
                    tags=["document", "notes"],
                ),
            ]
        )

        self.assertEqual(added_count, 2)
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(self.store.size(), 2)

    def test_delete_record_rebuilds_store_without_removed_fact(self) -> None:
        _, record = self.store.add_text("Rock music is my favorite.")
        self.store.add_text("I also enjoy classical music.")

        deleted = self.store.delete_record(record.id)

        self.assertTrue(deleted)
        self.assertEqual(self.store.facts(), ["I also enjoy classical music."])

    def test_update_record_rebuilds_store_with_new_text(self) -> None:
        _, record = self.store.add_text("Rock music is my favorite.")
        self.store.add_text("I also enjoy classical music.")

        status = self.store.update_record(
            record.id,
            "My favorite music is metal.",
            category="personal_context",
            source="chat",
            tags=["personal"],
        )

        self.assertEqual(status, "updated")
        self.assertEqual(
            self.store.facts(),
            ["My favorite music is metal.", "I also enjoy classical music."],
        )


if __name__ == "__main__":
    unittest.main()
