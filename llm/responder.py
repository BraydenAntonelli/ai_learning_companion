from __future__ import annotations

from typing import Dict, List, Optional

from memory.models import MemoryRecord, RetrievedMemory

DEFAULT_NO_ANSWER = "I couldn't find a relevant memory for that yet."
DEFAULT_LOW_CONFIDENCE = "I found something related, but I'm not confident enough to answer clearly."
DEFAULT_AMBIGUOUS_ANSWER = (
    "I found multiple similarly strong memories, so I don't want to guess."
)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def format_category_label(category: str) -> str:
    return category.replace("_", " ").title()


def _compute_confidence_score(
    best_score: float,
    min_similarity: float,
    score_gap: Optional[float],
    min_score_gap: float,
) -> int:
    upper_range = max(1.0 - min_similarity, 1e-6)
    score_component = _clamp((best_score - min_similarity) / upper_range, 0.0, 1.0)

    if score_gap is None:
        gap_component = 1.0 if best_score >= min_similarity else 0.0
    elif min_score_gap <= 0:
        gap_component = 1.0
    else:
        gap_component = _clamp(score_gap / min_score_gap, 0.0, 1.0)

    score = round(((score_component * 0.75) + (gap_component * 0.25)) * 100)
    return int(_clamp(score, 0, 100))


def _confidence_label(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 75:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def build_teach_response(record: MemoryRecord, added: bool) -> str:
    category = format_category_label(record.category).lower()
    if added:
        return f"I'll remember that as {category}."
    return f"I already had that saved in memory as {category}."


def build_answer_response(
    results: List[RetrievedMemory],
    min_similarity: float = 0.45,
    min_score_gap: float = 0.05,
) -> Dict[str, object]:
    """Convert retrieval results into a UI-friendly answer response."""
    if not results:
        return {
            "found": False,
            "answer": DEFAULT_NO_ANSWER,
            "score": None,
            "second_score": None,
            "score_gap": None,
            "source_record": None,
            "alternate_source_record": None,
            "confidence_score": None,
            "confidence_label": None,
            "rejection_reason": "no_results",
        }

    best_match = results[0]
    second_match: Optional[RetrievedMemory] = None
    if len(results) > 1:
        second_match = results[1]

    score_gap = None
    if second_match is not None:
        score_gap = best_match.score - second_match.score

    confidence_score = _compute_confidence_score(
        best_match.score,
        min_similarity=min_similarity,
        score_gap=score_gap,
        min_score_gap=min_score_gap,
    )

    rejection_reason: Optional[str] = None
    found = True
    answer = best_match.record.text

    if best_match.score < min_similarity:
        found = False
        answer = DEFAULT_LOW_CONFIDENCE
        rejection_reason = "low_confidence"
    elif score_gap is not None and score_gap < min_score_gap:
        found = False
        answer = DEFAULT_AMBIGUOUS_ANSWER
        rejection_reason = "ambiguous"
        confidence_score = min(confidence_score, 39)

    return {
        "found": found,
        "answer": answer,
        "score": best_match.score,
        "second_score": None if second_match is None else second_match.score,
        "score_gap": score_gap,
        "source_record": best_match.record,
        "alternate_source_record": None if second_match is None else second_match.record,
        "confidence_score": confidence_score,
        "confidence_label": _confidence_label(confidence_score),
        "rejection_reason": rejection_reason,
    }
