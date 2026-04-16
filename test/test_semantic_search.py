import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.vector_store import VectorStore
from retriever.semantic_search import search_memory


class SemanticSearchTests(unittest.TestCase):
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

    def test_empty_query_returns_no_results(self) -> None:
        self.assertEqual(search_memory("   ", self.store, top_k=1), [])

    def test_empty_store_returns_no_results_without_embedding(self) -> None:
        with patch("retriever.semantic_search.embed_text") as mock_embed:
            results = search_memory("What is my favorite music?", self.store, top_k=1)

        mock_embed.assert_not_called()
        self.assertEqual(results, [])

    def test_search_memory_embeds_normalized_query_and_returns_match(self) -> None:
        self.store.add([1.0, 0.0, 0.0], "My favorite music is rock.")
        self.store.add([0.0, 1.0, 0.0], "I also like jazz.")

        with patch(
            "retriever.semantic_search.embed_text",
            return_value=[0.9, 0.1, 0.0],
        ) as mock_embed:
            results = search_memory("  What is   my favorite music?  ", self.store, top_k=1)

        mock_embed.assert_called_once_with("What is my favorite music?")
        self.assertEqual(results[0].record.text, "My favorite music is rock.")


if __name__ == "__main__":
    unittest.main()
