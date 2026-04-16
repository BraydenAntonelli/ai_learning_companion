from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Protocol

from utils.paths import DEFAULT_CHUNK_SIZE
from utils.text_utils import chunk_text, normalize_text


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


@dataclass
class DocumentIngestionPlan:
    name: str
    source: str
    text: str
    chunks: List[str]
    tags: List[str]


def _filename_tags(name: str) -> List[str]:
    stem = Path(name).stem.casefold().replace("_", "-").replace(" ", "-")
    tags = ["document"]
    if stem:
        tags.append(stem)
    return tags


def extract_text_from_upload(uploaded_file: UploadedFileLike) -> str:
    suffix = Path(uploaded_file.name).suffix.casefold()
    raw_bytes = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return normalize_text(raw_bytes.decode(encoding))
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode '{uploaded_file.name}' as text.")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                "PDF upload requires the 'pypdf' package to be installed."
            ) from exc

        reader = PdfReader(BytesIO(raw_bytes))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        cleaned = normalize_text(text)
        if not cleaned:
            raise ValueError(f"'{uploaded_file.name}' did not contain readable text.")
        return cleaned

    raise ValueError(
        f"Unsupported file type '{suffix or '[no extension]'}'. Use .txt, .md, or .pdf files."
    )


def build_document_ingestion_plan(
    uploaded_file: UploadedFileLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DocumentIngestionPlan:
    text = extract_text_from_upload(uploaded_file)
    chunks = chunk_text(text, max_chars=chunk_size)
    if not chunks:
        raise ValueError(f"'{uploaded_file.name}' did not contain any text to ingest.")

    return DocumentIngestionPlan(
        name=uploaded_file.name,
        source=f"upload:{uploaded_file.name}",
        text=text,
        chunks=chunks,
        tags=_filename_tags(uploaded_file.name),
    )
