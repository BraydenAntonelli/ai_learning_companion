import unittest
from urllib import error
from unittest.mock import patch

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from llm.ollama_client import (
    LocalLLMConfig,
    generate_grounded_answer,
    get_ollama_status,
)
from memory.models import MemoryRecord, RetrievedMemory


def make_match(text: str, score: float) -> RetrievedMemory:
    return RetrievedMemory(
        record=MemoryRecord.create(
            text=text,
            category="factual_statement",
            source="manual",
        ),
        score=score,
    )


class OllamaClientTests(unittest.TestCase):
    def test_get_ollama_status_reports_ready_when_model_is_installed(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")

        with patch(
            "llm.ollama_client._http_get_json",
            side_effect=[
                {"version": "0.12.6"},
                {"models": [{"name": "llama3.2:3b"}, {"name": "gemma3"}]},
            ],
        ):
            status = get_ollama_status(config)

        self.assertTrue(status.available)
        self.assertEqual(status.version, "0.12.6")
        self.assertIn("llama3.2:3b", status.model_names)

    def test_get_ollama_status_reports_missing_model(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")

        with patch(
            "llm.ollama_client._http_get_json",
            side_effect=[
                {"version": "0.12.6"},
                {"models": [{"name": "gemma3"}]},
            ],
        ):
            status = get_ollama_status(config)

        self.assertFalse(status.available)
        self.assertIn("ollama pull llama3.2:3b", status.message)

    def test_generate_grounded_answer_returns_chat_output(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")
        results = [make_match("My favorite music is rock.", 0.91)]

        with patch(
            "llm.ollama_client._http_post_json",
            return_value={
                "model": "llama3.2:3b",
                "message": {"content": "Your favorite music is rock."},
                "prompt_eval_count": 15,
                "eval_count": 8,
            },
        ):
            response = generate_grounded_answer(
                "What is my favorite music?",
                results,
                {"found": True, "confidence_score": 94, "rejection_reason": None},
                config,
            )

        self.assertTrue(response.used)
        self.assertEqual(response.answer, "Your favorite music is rock.")
        self.assertEqual(response.model, "llama3.2:3b")
        self.assertEqual(response.usage["eval_count"], 8)

    def test_generate_grounded_answer_handles_local_server_error(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")
        results = [make_match("My favorite music is rock.", 0.91)]

        with patch(
            "llm.ollama_client._http_post_json",
            side_effect=error.URLError("connection refused"),
        ):
            response = generate_grounded_answer(
                "What is my favorite music?",
                results,
                {"found": True, "confidence_score": 94, "rejection_reason": None},
                config,
            )

        self.assertFalse(response.used)
        self.assertIsNone(response.answer)
        self.assertIn("Couldn't reach Ollama", response.error)

    def test_generate_grounded_answer_rejects_meta_responses(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")
        results = [make_match("Pizza is my favorite food.", 0.91)]

        with patch(
            "llm.ollama_client._http_post_json",
            return_value={
                "model": "llama3.2:3b",
                "message": {
                    "content": (
                        "I'm not sure yet, based on the stored memory. "
                        "The confidence score seems low."
                    )
                },
            },
        ):
            response = generate_grounded_answer(
                "What is my favorite food?",
                results,
                {"found": True, "confidence_score": 88, "rejection_reason": None},
                config,
            )

        self.assertFalse(response.used)
        self.assertIsNone(response.answer)
        self.assertIn("meta answer", response.error)

    def test_generate_grounded_answer_skips_generation_for_rejected_matches(self) -> None:
        config = LocalLLMConfig(enabled=True, model="llama3.2:3b")
        results = [make_match("My favorite music is rock.", 0.31)]

        with patch("llm.ollama_client._http_post_json") as mock_post:
            response = generate_grounded_answer(
                "What is my favorite music?",
                results,
                {"found": False, "confidence_score": 22, "rejection_reason": "low_confidence"},
                config,
            )

        mock_post.assert_not_called()
        self.assertFalse(response.used)
        self.assertIsNone(response.answer)


if __name__ == "__main__":
    unittest.main()
