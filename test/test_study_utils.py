import unittest

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.models import MemoryRecord
from utils.study_utils import build_flashcard


class StudyUtilsTests(unittest.TestCase):
    def test_builds_date_question_from_historical_fact(self) -> None:
        record = MemoryRecord.create(
            text="The Battle of Hastings occurred in 1066.",
            category="factual_statement",
            source="manual",
        )

        flashcard = build_flashcard(record)

        self.assertEqual(flashcard.prompt, "When did The Battle of Hastings occur?")
        self.assertEqual(flashcard.answer, "The Battle of Hastings occurred in 1066.")

    def test_builds_definition_prompt_for_is_statements(self) -> None:
        record = MemoryRecord.create(
            text="Photosynthesis is the process plants use to convert sunlight into energy.",
            category="academic_concept",
            source="manual",
        )

        flashcard = build_flashcard(record)

        self.assertEqual(flashcard.prompt, "What is Photosynthesis?")


if __name__ == "__main__":
    unittest.main()
