import tempfile
import unittest
from pathlib import Path

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.store = VectorStore(
            dim=3,
            index_path=temp_path / "memory.faiss",
            metadata_path=temp_path / "memory.json",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_and_search_return_closest_match(self) -> None:
        self.assertTrue(self.store.add([0.0, 0.0, 0.0], "Rock music is my favorite."))
        self.assertTrue(self.store.add([5.0, 5.0, 5.0], "I also enjoy classical music."))

        results = self.store.search([0.1, 0.0, 0.0], top_k=2)

        self.assertEqual(results[0][0], "Rock music is my favorite.")
        self.assertLess(results[0][1], results[1][1])

    def test_duplicate_detection_ignores_case_and_extra_spacing(self) -> None:
        self.assertTrue(self.store.add([0.0, 0.0, 0.0], "My favorite music is rock."))
        self.assertFalse(self.store.add([1.0, 1.0, 1.0], "  my favorite music is rock.  "))

    def test_delete_fact_rebuilds_store_without_removed_fact(self) -> None:
        self.store.add([0.0, 0.0, 0.0], "My favorite music is rock.")
        self.store.add([5.0, 5.0, 5.0], "I also enjoy classical music.")

        embedding_map = {
            "I also enjoy classical music.": [5.0, 5.0, 5.0],
        }

        deleted = self.store.delete_fact(
            "My favorite music is rock.",
            lambda text: embedding_map[text],
        )

        self.assertTrue(deleted)
        self.assertEqual(self.store.facts(), ["I also enjoy classical music."])

    def test_update_fact_rebuilds_store_with_new_text(self) -> None:
        self.store.add([0.0, 0.0, 0.0], "My favorite music is rock.")
        self.store.add([5.0, 5.0, 5.0], "I also enjoy classical music.")

        embedding_map = {
            "My favorite music is metal.": [0.1, 0.0, 0.0],
            "I also enjoy classical music.": [5.0, 5.0, 5.0],
        }

        status = self.store.update_fact(
            "My favorite music is rock.",
            "My favorite music is metal.",
            lambda text: embedding_map[text],
        )

        self.assertEqual(status, "updated")
        self.assertEqual(
            self.store.facts(),
            ["My favorite music is metal.", "I also enjoy classical music."],
        )


if __name__ == "__main__":
    unittest.main()
