import unittest

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from utils.document_utils import build_document_ingestion_plan, split_document_text


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class DocumentUtilsTests(unittest.TestCase):
    def test_builds_ingestion_plan_for_text_file(self) -> None:
        uploaded_file = FakeUploadedFile(
            "notes.txt",
            b"Line one with useful context.\nLine two with more useful context.",
        )

        plan = build_document_ingestion_plan(uploaded_file, chunk_size=30)

        self.assertEqual(plan.source, "upload:notes.txt")
        self.assertGreaterEqual(len(plan.chunks), 2)
        self.assertIn("document", plan.tags)
        self.assertIn("notes", plan.tags)

    def test_rejects_unsupported_file_types(self) -> None:
        uploaded_file = FakeUploadedFile("notes.csv", b"a,b,c")

        with self.assertRaises(ValueError):
            build_document_ingestion_plan(uploaded_file)

    def test_split_document_text_keeps_one_fact_per_line_when_possible(self) -> None:
        chunks = split_document_text(
            "Mercury is the closest planet to the Sun.\nVenus is the hottest planet.\nEarth has one moon.",
            max_chars=120,
        )

        self.assertEqual(
            chunks,
            [
                "Mercury is the closest planet to the Sun.",
                "Venus is the hottest planet.",
                "Earth has one moon.",
            ],
        )

    def test_split_document_text_handles_bullets_and_numbered_lists(self) -> None:
        chunks = split_document_text(
            "- Rock is a genre of music.\n- Jazz uses improvisation.\n1. Blues influenced rock.",
            max_chars=120,
        )

        self.assertEqual(
            chunks,
            [
                "Rock is a genre of music.",
                "Jazz uses improvisation.",
                "Blues influenced rock.",
            ],
        )

    def test_split_document_text_applies_heading_context(self) -> None:
        chunks = split_document_text(
            "# Biology Facts\nCells are the basic unit of life.\nPhotosynthesis happens in chloroplasts.",
            max_chars=120,
        )

        self.assertEqual(
            chunks,
            [
                "Biology Facts: Cells are the basic unit of life.",
                "Biology Facts: Photosynthesis happens in chloroplasts.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
