from __future__ import annotations

from dataclasses import dataclass

from memory.models import MemoryRecord


@dataclass
class Flashcard:
    prompt: str
    answer: str
    hint: str


def build_flashcard(record: MemoryRecord) -> Flashcard:
    text = record.text.rstrip(".")
    lowered = text.casefold()

    if " occurred in " in lowered:
        subject, _, _ = text.partition(" occurred in ")
        prompt = f"When did {subject} occur?"
        hint = "Focus on the date or time reference."
    elif lowered.startswith("my ") and " is " in lowered:
        subject, _, _ = text.partition(" is ")
        prompt = f"What is your {subject[3:]}?"
        hint = "Recall the personal detail being stored."
    elif " is " in lowered:
        subject, _, _ = text.partition(" is ")
        prompt = f"What is {subject}?"
        hint = "Try to restate the definition or fact."
    elif " are " in lowered:
        subject, _, _ = text.partition(" are ")
        prompt = f"What are {subject}?"
        hint = "Recall the related fact."
    else:
        prompt = "What should you remember about this card?"
        hint = "Try to explain the main idea in your own words."

    return Flashcard(
        prompt=prompt,
        answer=record.text,
        hint=hint,
    )
