import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from feedback.logger import clear_feedback_log, get_feedback_stats, log_feedback


class FeedbackLoggerTests(unittest.TestCase):
    def test_feedback_is_logged_counted_and_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback_log.jsonl"

            with patch("feedback.logger.FEEDBACK_LOG_PATH", feedback_path):
                log_feedback(
                    question="What is my favorite music?",
                    answer="My favorite music is rock.",
                    label="up",
                    score=0.92,
                    source_record_id="abc123",
                    source_text="My favorite music is rock.",
                    source_category="personal_context",
                    confidence_score=92,
                    rejection_reason=None,
                )
                log_feedback(
                    question="What is my favorite color?",
                    answer="I do not know yet.",
                    label="down",
                    score=0.18,
                    source_record_id="def456",
                    source_text="My favorite color is blue.",
                    source_category="personal_context",
                    confidence_score=18,
                    rejection_reason="low_confidence",
                )

                self.assertEqual(
                    get_feedback_stats(),
                    {"total": 2, "up": 1, "down": 1},
                )

                records = [
                    json.loads(line)
                    for line in feedback_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(records[0]["score"], 0.92)
                self.assertEqual(records[1]["rejection_reason"], "low_confidence")

                clear_feedback_log()
                self.assertFalse(feedback_path.exists())
                self.assertEqual(
                    get_feedback_stats(),
                    {"total": 0, "up": 0, "down": 0},
                )


if __name__ == "__main__":
    unittest.main()
