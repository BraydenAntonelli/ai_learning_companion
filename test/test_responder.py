import unittest

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from llm.responder import (
    DEFAULT_AMBIGUOUS_ANSWER,
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_NO_ANSWER,
    build_answer_response,
)
from memory.models import MemoryRecord, RetrievedMemory


def make_record(text: str) -> MemoryRecord:
    return MemoryRecord.create(
        text=text,
        category="factual_statement",
        source="manual",
    )


class BuildAnswerResponseTests(unittest.TestCase):
    def test_returns_no_answer_when_results_are_empty(self) -> None:
        response = build_answer_response([], min_similarity=0.45)

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_NO_ANSWER)
        self.assertIsNone(response["score"])
        self.assertEqual(response["rejection_reason"], "no_results")

    def test_returns_best_answer_when_match_is_within_cutoff(self) -> None:
        response = build_answer_response(
            [RetrievedMemory(make_record("My favorite music is rock."), 0.92)],
            min_similarity=0.45,
            min_score_gap=0.05,
        )

        self.assertTrue(response["found"])
        self.assertEqual(response["answer"], "My favorite music is rock.")
        self.assertEqual(response["score"], 0.92)
        self.assertEqual(response["source_record"].text, "My favorite music is rock.")
        self.assertGreaterEqual(response["confidence_score"], 75)

    def test_returns_low_confidence_fallback_when_match_is_too_weak(self) -> None:
        response = build_answer_response(
            [RetrievedMemory(make_record("My favorite music is rock."), 0.18)],
            min_similarity=0.45,
            min_score_gap=0.05,
        )

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_LOW_CONFIDENCE)
        self.assertEqual(response["score"], 0.18)
        self.assertEqual(response["rejection_reason"], "low_confidence")

    def test_rejects_ambiguous_matches_when_top_two_results_are_too_close(self) -> None:
        response = build_answer_response(
            [
                RetrievedMemory(make_record("My favorite music is rock."), 0.78),
                RetrievedMemory(make_record("My favorite music is jazz."), 0.74),
            ],
            min_similarity=0.45,
            min_score_gap=0.05,
        )

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_AMBIGUOUS_ANSWER)
        self.assertEqual(response["rejection_reason"], "ambiguous")
        self.assertEqual(response["alternate_source_record"].text, "My favorite music is jazz.")
        self.assertLess(response["confidence_score"], 40)


if __name__ == "__main__":
    unittest.main()
