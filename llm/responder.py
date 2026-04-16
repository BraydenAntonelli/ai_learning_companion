from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SearchResult = Tuple[str, float]

DEFAULT_NO_ANSWER = "I'm sorry, I don't seem to have a clear answer for that one."
DEFAULT_LOW_CONFIDENCE = "I'm not confident enough to answer that yet. Try teaching me first."
DEFAULT_AMBIGUOUS_ANSWER = (
    "I found more than one close memory and I'm not confident which one you mean yet."
)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _compute_confidence_score(
    best_distance: float,
    max_distance: float,
    distance_gap: Optional[float],
    min_distance_gap: float,
) -> int:
    if max_distance <= 0:
        return 0

    distance_component = _clamp(1 - (best_distance / max_distance), 0.0, 1.0)
    if distance_gap is None:
        gap_component = 1.0 if best_distance <= max_distance else 0.0
    elif min_distance_gap <= 0:
        gap_component = 1.0
    else:
        gap_component = _clamp(distance_gap / min_distance_gap, 0.0, 1.0)

    score = round(((distance_component * 0.75) + (gap_component * 0.25)) * 100)
    return int(_clamp(score, 0, 100))


def _confidence_label(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 75:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def build_answer_response(
    results: List[SearchResult],
    max_distance: float = 1.0,
    min_distance_gap: float = 0.15,
) -> Dict[str, object]:
    """Convert retrieval results into a UI-friendly answer response."""
    if not results:
        return {
            "found": False,
            "answer": DEFAULT_NO_ANSWER,
            "distance": None,
            "second_distance": None,
            "distance_gap": None,
            "source_text": None,
            "alternate_source_text": None,
            "confidence_score": None,
            "confidence_label": None,
            "rejection_reason": "no_results",
        }

    best_text, best_distance = results[0]
    second_text: Optional[str] = None
    second_distance: Optional[float] = None
    if len(results) > 1:
        second_text, second_distance = results[1]

    distance_gap = None
    if second_distance is not None:
        distance_gap = second_distance - best_distance

    confidence_score = _compute_confidence_score(
        best_distance,
        max_distance=max_distance,
        distance_gap=distance_gap,
        min_distance_gap=min_distance_gap,
    )

    rejection_reason: Optional[str] = None
    found = True
    answer = best_text

    if best_distance > max_distance:
        found = False
        answer = DEFAULT_LOW_CONFIDENCE
        rejection_reason = "low_confidence"
    elif distance_gap is not None and distance_gap < min_distance_gap:
        found = False
        answer = DEFAULT_AMBIGUOUS_ANSWER
        rejection_reason = "ambiguous"
        confidence_score = min(confidence_score, 39)

    return {
        "found": found,
        "answer": answer,
        "distance": best_distance,
        "second_distance": second_distance,
        "distance_gap": distance_gap,
        "source_text": best_text,
        "alternate_source_text": second_text,
        "confidence_score": confidence_score,
        "confidence_label": _confidence_label(confidence_score),
        "rejection_reason": rejection_reason,
    }
