import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

"""Smoke tests the basic teach-then-ask flow."""

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from llm.responder import DEFAULT_LOW_CONFIDENCE, build_answer_response
from memory.vector_store import VectorStore
from retriever.semantic_search import search_memory


class TeachAndAskFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.store = VectorStore(
            dim=3,
            index_path=temp_path / "memory.faiss",
            metadata_path=temp_path / "memory.sqlite3",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_teach_then_ask_returns_answer_when_similarity_is_within_cutoff(self) -> None:
        self.store.add([1.0, 0.0, 0.0], "My favorite music is rock.")

        with patch(
            "retriever.semantic_search.embed_texts",
            return_value=[
                [0.9, 0.1, 0.0],
                [0.9, 0.1, 0.0],
                [0.9, 0.1, 0.0],
            ],
        ):
            results = search_memory("What is my favorite music?", self.store, top_k=1)

        response = build_answer_response(results, min_similarity=0.7)

        self.assertTrue(response["found"])
        self.assertEqual(response["answer"], "My favorite music is rock.")

    def test_low_similarity_match_returns_fallback_response(self) -> None:
        self.store.add([0.0, 1.0, 0.0], "My favorite music is rock.")

        with patch(
            "retriever.semantic_search.embed_texts",
            return_value=[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ):
            results = search_memory("What is my favorite music?", self.store, top_k=1)

        response = build_answer_response(results, min_similarity=0.7)

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_LOW_CONFIDENCE)
        self.assertIsNotNone(response["score"])


if __name__ == "__main__":
    unittest.main()
