from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_INDEX_PATH = DATA_DIR / "memory.faiss"
MEMORY_METADATA_PATH = DATA_DIR / "memory.json"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 500
SUPPORTED_UPLOAD_EXTENSIONS = ("txt", "md", "pdf")
