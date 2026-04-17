from __future__ import annotations

"""Helpers for turning uploaded files into memory-friendly chunks."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
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


LIST_ITEM_PATTERN = re.compile(r"^(?:[-*+•]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)")
MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _filename_tags(name: str) -> List[str]:
    stem = Path(name).stem.casefold().replace("_", "-").replace(" ", "-")
    tags = ["document"]
    if stem:
        tags.append(stem)
    return tags


def _normalize_document_text(text: str) -> str:
    # Keep paragraph breaks, but clean up messy whitespace inside each line.
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        lines.append(line)

    normalized_lines: List[str] = []
    last_was_blank = False
    for line in lines:
        if not line:
            if normalized_lines and not last_was_blank:
                normalized_lines.append("")
            last_was_blank = True
            continue

        normalized_lines.append(line)
        last_was_blank = False

    return "\n".join(normalized_lines).strip()


def _strip_list_prefix(line: str) -> str:
    return LIST_ITEM_PATTERN.sub("", line, count=1).strip()


def _is_heading_line(line: str) -> bool:
    if MARKDOWN_HEADING_PATTERN.match(line):
        return True
    return line.endswith(":") and len(line) <= 80 and len(line.split()) <= 10


def _strip_heading_prefix(line: str) -> str:
    if MARKDOWN_HEADING_PATTERN.match(line):
        return line.lstrip("#").strip()
    return line[:-1].strip() if line.endswith(":") else line.strip()


def _with_heading(text: str, heading: str | None) -> str:
    if not heading:
        return normalize_text(text)
    return normalize_text(f"{heading}: {text}")


def _split_long_unit(text: str, max_chars: int) -> List[str]:
    # Try sentence-aware splits first, then fall back to the plain chunker if needed.
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [normalize_text(sentence) for sentence in SENTENCE_SPLIT_PATTERN.split(cleaned)]
    sentences = [sentence for sentence in sentences if sentence]
    if len(sentences) <= 1:
        return chunk_text(cleaned, max_chars=max_chars)

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.extend(chunk_text(sentence, max_chars=max_chars))
            continue

        additional_length = len(sentence) if not current_sentences else len(sentence) + 1
        if current_length + additional_length > max_chars:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_length = len(sentence)
        else:
            current_sentences.append(sentence)
            current_length += additional_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def split_document_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    # This is where line-based facts, bullets, and paragraph notes all get normalized into cleaner units.
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    cleaned = _normalize_document_text(text)
    if not cleaned:
        return []

    units: List[str] = []
    paragraph_lines: List[str] = []
    current_heading: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        units.append(_with_heading(" ".join(paragraph_lines), current_heading))
        paragraph_lines = []

    for line in cleaned.split("\n"):
        if not line:
            flush_paragraph()
            continue

        if _is_heading_line(line):
            flush_paragraph()
            current_heading = _strip_heading_prefix(line)
            continue

        if LIST_ITEM_PATTERN.match(line):
            flush_paragraph()
            units.append(_with_heading(_strip_list_prefix(line), current_heading))
            continue

        paragraph_lines.append(line)
        if line.endswith((".", "!", "?")):
            flush_paragraph()

    flush_paragraph()

    chunks: List[str] = []
    for unit in units:
        chunks.extend(_split_long_unit(unit, max_chars=max_chars))

    return chunks


def extract_text_from_upload(uploaded_file: UploadedFileLike) -> str:
    suffix = Path(uploaded_file.name).suffix.casefold()
    raw_bytes = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return _normalize_document_text(raw_bytes.decode(encoding))
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
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        cleaned = _normalize_document_text(text)
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
    # The plan keeps the raw extracted text plus the final chunks the store will save.
    text = extract_text_from_upload(uploaded_file)
    chunks = split_document_text(text, max_chars=chunk_size)
    if not chunks:
        raise ValueError(f"'{uploaded_file.name}' did not contain any text to ingest.")

    return DocumentIngestionPlan(
        name=uploaded_file.name,
        source=f"upload:{uploaded_file.name}",
        text=text,
        chunks=chunks,
        tags=_filename_tags(uploaded_file.name),
    )
