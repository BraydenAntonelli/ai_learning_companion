import unittest

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from llm.responder import (
    DEFAULT_AMBIGUOUS_ANSWER,
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_NO_ANSWER,
    build_answer_response,
)


class BuildAnswerResponseTests(unittest.TestCase):
    def test_returns_no_answer_when_results_are_empty(self) -> None:
        response = build_answer_response([], max_distance=1.0)

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_NO_ANSWER)
        self.assertIsNone(response["distance"])
        self.assertEqual(response["rejection_reason"], "no_results")

    def test_returns_best_answer_when_match_is_within_cutoff(self) -> None:
        response = build_answer_response(
            [("My favorite music is rock.", 0.12)],
            max_distance=0.5,
            min_distance_gap=0.15,
        )

        self.assertTrue(response["found"])
        self.assertEqual(response["answer"], "My favorite music is rock.")
        self.assertEqual(response["distance"], 0.12)
        self.assertEqual(response["source_text"], "My favorite music is rock.")
        self.assertGreaterEqual(response["confidence_score"], 75)

    def test_returns_low_confidence_fallback_when_match_is_too_far(self) -> None:
        response = build_answer_response(
            [("My favorite music is rock.", 1.4)],
            max_distance=0.5,
            min_distance_gap=0.15,
        )

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_LOW_CONFIDENCE)
        self.assertEqual(response["distance"], 1.4)
        self.assertEqual(response["rejection_reason"], "low_confidence")

    def test_rejects_ambiguous_matches_when_top_two_results_are_too_close(self) -> None:
        response = build_answer_response(
            [
                ("My favorite music is rock.", 0.18),
                ("My favorite music is jazz.", 0.23),
            ],
            max_distance=0.5,
            min_distance_gap=0.10,
        )

        self.assertFalse(response["found"])
        self.assertEqual(response["answer"], DEFAULT_AMBIGUOUS_ANSWER)
        self.assertEqual(response["rejection_reason"], "ambiguous")
        self.assertEqual(response["alternate_source_text"], "My favorite music is jazz.")
        self.assertLess(response["confidence_score"], 40)


if __name__ == "__main__":
    unittest.main()
