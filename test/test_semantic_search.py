import unittest
from unittest.mock import patch

"""Makes sure query cleanup still gives semantic search a fair shot."""

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.models import MemoryRecord, RetrievedMemory
from retriever.semantic_search import search_memory


class SemanticSearchTests(unittest.TestCase):
    def _make_store(self, result_map: dict[tuple[float, ...], list[RetrievedMemory]]):
        class FakeStore:
            def __init__(self, result_map: dict[tuple[float, ...], list[RetrievedMemory]]) -> None:
                self.result_map = result_map

            def size(self) -> int:
                return 0 if not self.result_map else 1

            def search(self, embedding: list[float], top_k: int = 1) -> list[RetrievedMemory]:
                return list(self.result_map.get(tuple(embedding), []))[:top_k]

        return FakeStore(result_map)

    def test_empty_query_returns_no_results(self) -> None:
        store = self._make_store({})
        self.assertEqual(search_memory("   ", store, top_k=1), [])

    def test_empty_store_returns_no_results_without_embedding(self) -> None:
        store = self._make_store({})
        with patch("retriever.semantic_search.embed_texts") as mock_embed:
            results = search_memory("What is my favorite music?", store, top_k=1)

        mock_embed.assert_not_called()
        self.assertEqual(results, [])

    def test_search_memory_embeds_normalized_query_and_returns_match(self) -> None:
        store = self._make_store(
            {
                (0.9, 0.1, 0.0): [
                    RetrievedMemory(
                        MemoryRecord.create(
                            text="My favorite music is rock.",
                            category="personal_context",
                            source="manual",
                        ),
                        0.92,
                    )
                ]
            }
        )

        with patch(
            "retriever.semantic_search.embed_texts",
            return_value=[
                [0.9, 0.1, 0.0],
                [0.9, 0.1, 0.0],
                [0.9, 0.1, 0.0],
            ],
        ) as mock_embed:
            results = search_memory("  What is   my favorite music?  ", store, top_k=1)

        embedded_queries = mock_embed.call_args.args[0]
        self.assertIn("What is my favorite music", embedded_queries)
        self.assertIn("my favorite music", embedded_queries)
        self.assertEqual(results[0].record.text, "My favorite music is rock.")

    def test_search_memory_tries_cleaner_query_variants_for_preference_questions(self) -> None:
        store = self._make_store(
            {
                (0.0, 1.0, 0.0): [
                    RetrievedMemory(
                        MemoryRecord.create(
                            text="The capital of Florida is Tallahassee.",
                            category="factual_statement",
                            source="manual",
                        ),
                        0.45,
                    )
                ],
                (1.0, 0.0, 0.0): [
                    RetrievedMemory(
                        MemoryRecord.create(
                            text="Pizza is my favorite food.",
                            category="personal_context",
                            source="manual",
                        ),
                        0.88,
                    )
                ],
            }
        )

        def fake_embed_many(queries: list[str]) -> list[list[float]]:
            embeddings: list[list[float]] = []
            for query in queries:
                if query == "What is my favorite food again":
                    embeddings.append([0.0, 1.0, 0.0])
                elif query == "my favorite food":
                    embeddings.append([1.0, 0.0, 0.0])
                elif query == "favorite food":
                    embeddings.append([1.0, 0.0, 0.0])
                else:
                    embeddings.append([0.0, 0.0, 1.0])
            return embeddings

        with patch("retriever.semantic_search.embed_texts", side_effect=fake_embed_many) as mock_embed:
            results = search_memory("What is my favorite food again?", store, top_k=1)

        self.assertEqual(results[0].record.text, "Pizza is my favorite food.")
        embedded_queries = mock_embed.call_args.args[0]
        self.assertIn("my favorite food", embedded_queries)


if __name__ == "__main__":
    unittest.main()
