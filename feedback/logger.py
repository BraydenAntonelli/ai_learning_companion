from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Optional

from utils.paths import FEEDBACK_DB_PATH

CREATE_FEEDBACK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL,
    source_record_id TEXT,
    source_text TEXT,
    source_category TEXT,
    confidence_score INTEGER,
    rejection_reason TEXT
)
"""


def _connect() -> sqlite3.Connection:
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(FEEDBACK_DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
def _init_feedback_store() -> None:
    with _connection() as connection:
        connection.execute(CREATE_FEEDBACK_TABLE_SQL)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_label ON feedback(label)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)"
        )


def log_feedback(
    question: str,
    answer: str,
    label: str,
    score: Optional[float] = None,
    source_record_id: Optional[str] = None,
    source_text: Optional[str] = None,
    source_category: Optional[str] = None,
    confidence_score: Optional[int] = None,
    rejection_reason: Optional[str] = None,
) -> None:
    _init_feedback_store()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                timestamp,
                question,
                answer,
                label,
                score,
                source_record_id,
                source_text,
                source_category,
                confidence_score,
                rejection_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                answer,
                label,
                score,
                source_record_id,
                source_text,
                source_category,
                confidence_score,
                rejection_reason,
            ),
        )


def get_feedback_stats() -> Dict[str, int]:
    _init_feedback_store()
    stats = {"total": 0, "up": 0, "down": 0}

    with _connection() as connection:
        for row in connection.execute(
            "SELECT label, COUNT(*) AS count FROM feedback GROUP BY label"
        ):
            label = row["label"]
            count = int(row["count"])
            stats["total"] += count
            if label == "up":
                stats["up"] = count
            elif label == "down":
                stats["down"] = count

    return stats


def clear_feedback_log() -> None:
    _init_feedback_store()
    with _connection() as connection:
        connection.execute("DELETE FROM feedback")
