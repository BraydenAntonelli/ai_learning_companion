from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional

from utils.paths import FEEDBACK_LOG_PATH


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
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "label": label,
        "score": score,
        "source_record_id": source_record_id,
        "source_text": source_text,
        "source_category": source_category,
        "confidence_score": confidence_score,
        "rejection_reason": rejection_reason,
    }
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_feedback_stats() -> Dict[str, int]:
    stats = {"total": 0, "up": 0, "down": 0}
    if not FEEDBACK_LOG_PATH.exists():
        return stats

    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total"] += 1
            label = record.get("label")
            if label == "up":
                stats["up"] += 1
            elif label == "down":
                stats["down"] += 1

    return stats


def clear_feedback_log() -> None:
    if FEEDBACK_LOG_PATH.exists():
        FEEDBACK_LOG_PATH.unlink()
