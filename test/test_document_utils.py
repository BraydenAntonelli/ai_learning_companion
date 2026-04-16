import unittest

from support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from utils.document_utils import build_document_ingestion_plan


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


if __name__ == "__main__":
    unittest.main()
