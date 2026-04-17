from __future__ import annotations

"""Central place for repo-local data paths and shared constants."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_INDEX_PATH = DATA_DIR / "memory.faiss"
MEMORY_DB_PATH = DATA_DIR / "memory.sqlite3"
MEMORY_METADATA_PATH = MEMORY_DB_PATH
FEEDBACK_DB_PATH = MEMORY_DB_PATH
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 500
SUPPORTED_UPLOAD_EXTENSIONS = ("txt", "md", "pdf")
