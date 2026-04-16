import unittest
from unittest.mock import Mock, patch

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory import embedder
from memory.embedder import embed_text, embed_texts


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

    def astype(self, _dtype: str):
        return self

    def tolist(self):
        return list(self.values)


class EmbedTextTests(unittest.TestCase):
    def setUp(self) -> None:
        embedder.get_model.cache_clear()

    def tearDown(self) -> None:
        embedder.get_model.cache_clear()

    def test_embed_text_normalizes_input_before_encoding(self) -> None:
        fake_model = Mock()
        fake_model.encode.return_value = FakeEmbedding([[1.0, 2.0, 3.0]])

        with patch("memory.embedder.get_model", return_value=fake_model):
            result = embed_text("  The   dog sat on the lawn.  ")

        fake_model.encode.assert_called_once_with(
            ["The dog sat on the lawn."],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_embed_texts_handles_multiple_inputs(self) -> None:
        fake_model = Mock()
        fake_model.encode.return_value = FakeEmbedding([[1.0, 2.0], [3.0, 4.0]])

        with patch("memory.embedder.get_model", return_value=fake_model):
            results = embed_texts(["  one  ", "two"])

        self.assertEqual(results, [[1.0, 2.0], [3.0, 4.0]])

    def test_embed_text_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            embed_text("   ")

    def test_get_model_falls_back_when_transformer_load_fails(self) -> None:
        with patch(
            "memory.embedder._load_sentence_transformer_model",
            side_effect=NotImplementedError("Cannot copy out of meta tensor; no data!"),
        ):
            model = embedder.get_model()

        self.assertIsInstance(model, embedder._HashingFallbackModel)
        self.assertIsNotNone(embedder.get_fallback_reason())

    def test_hashing_fallback_returns_expected_dimension(self) -> None:
        fallback_model = embedder._HashingFallbackModel()

        with patch("memory.embedder.get_model", return_value=fallback_model):
            result = embed_text("remember my favorite music is rock")

        self.assertEqual(len(result), 384)
        self.assertTrue(any(value != 0.0 for value in result))


if __name__ == "__main__":
    unittest.main()
