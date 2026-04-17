import unittest

"""Checks the rule-based classifier so the obvious categories stay obvious."""

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from memory.classifier import (
    classify_memory_text,
    detect_message_intent,
    looks_like_question,
    strip_teach_prefix,
)


class MemoryClassifierTests(unittest.TestCase):
    def test_detects_question_like_input(self) -> None:
        self.assertTrue(looks_like_question("What is my favorite music?"))
        self.assertEqual(detect_message_intent("How do plants get energy?"), "ask")

    def test_strips_teach_prefixes(self) -> None:
        self.assertEqual(
            strip_teach_prefix("Remember that my favorite music is rock."),
            "my favorite music is rock.",
        )

    def test_classifies_personal_context(self) -> None:
        classification = classify_memory_text("My favorite music is rock.")
        self.assertEqual(classification.category, "personal_context")

    def test_classifies_embedded_personal_preference_as_personal_context(self) -> None:
        classification = classify_memory_text("Pizza is my favorite food.")
        self.assertEqual(classification.category, "personal_context")

    def test_classifies_academic_concept(self) -> None:
        classification = classify_memory_text("Photosynthesis is the process plants use to convert sunlight into energy.")
        self.assertEqual(classification.category, "academic_concept")

    def test_classifies_document_excerpt_by_source(self) -> None:
        classification = classify_memory_text(
            "This is a document chunk.",
            source="upload:biology_notes.pdf",
        )
        self.assertEqual(classification.category, "document_excerpt")
        self.assertIn("document", classification.tags)


if __name__ == "__main__":
    unittest.main()
