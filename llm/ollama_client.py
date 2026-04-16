from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, request

from memory.models import RetrievedMemory
from utils.text_utils import normalize_text

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_TIMEOUT_SECONDS = 30.0
META_ANSWER_MARKERS = (
    "retrieval summary",
    "confidence score",
    "similarity:",
    "based on the stored memory",
    "stored memory",
)


@dataclass(frozen=True)
class LocalLLMConfig:
    enabled: bool = False
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/") or DEFAULT_OLLAMA_BASE_URL

    @property
    def normalized_model(self) -> str:
        return normalize_text(self.model) or DEFAULT_OLLAMA_MODEL


@dataclass(frozen=True)
class LocalLLMStatus:
    available: bool
    message: str
    model_names: List[str] = field(default_factory=list)
    version: Optional[str] = None


@dataclass(frozen=True)
class LocalLLMResult:
    used: bool
    answer: Optional[str]
    model: str
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def _read_json_response(response: Any) -> Dict[str, Any]:
    payload = response.read().decode("utf-8")
    return json.loads(payload)


def _http_get_json(url: str, timeout_seconds: float) -> Dict[str, Any]:
    with request.urlopen(url, timeout=timeout_seconds) as response:
        return _read_json_response(response)


def _http_post_json(url: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return _read_json_response(response)


def _extract_error_message(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        try:
            payload = exc.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])
        except Exception:
            pass
        return f"Ollama returned HTTP {exc.code}."

    if isinstance(exc, error.URLError):
        return "Couldn't reach Ollama at the configured local URL."

    return str(exc) or "Local LLM request failed."


def get_ollama_status(config: LocalLLMConfig) -> LocalLLMStatus:
    base_url = config.normalized_base_url

    try:
        version_payload = _http_get_json(
            f"{base_url}/api/version",
            timeout_seconds=min(config.timeout_seconds, 2.0),
        )
        tags_payload = _http_get_json(
            f"{base_url}/api/tags",
            timeout_seconds=min(config.timeout_seconds, 2.0),
        )
    except Exception:
        return LocalLLMStatus(
            available=False,
            message=(
                "Ollama does not look available yet. Start Ollama locally and make sure "
                f"it is serving at {base_url}."
            ),
            version=None,
        )

    raw_models = tags_payload.get("models", [])
    model_names = [
        str(model.get("name"))
        for model in raw_models
        if isinstance(model, dict) and model.get("name")
    ]
    version = None if not isinstance(version_payload, dict) else version_payload.get("version")

    if config.normalized_model not in model_names:
        if model_names:
            return LocalLLMStatus(
                available=False,
                message=(
                    f"Ollama is running, but `{config.normalized_model}` is not installed yet. "
                    f"Pull it locally with `ollama pull {config.normalized_model}`."
                ),
                model_names=model_names,
                version=None if version is None else str(version),
            )
        return LocalLLMStatus(
            available=False,
            message=(
                "Ollama is running, but no local models are installed yet. "
                f"Try `ollama pull {config.normalized_model}`."
            ),
            version=None if version is None else str(version),
        )

    return LocalLLMStatus(
        available=True,
        message=f"Ollama is ready with `{config.normalized_model}`.",
        model_names=model_names,
        version=None if version is None else str(version),
    )


def _build_grounding_messages(
    question: str,
    results: List[RetrievedMemory],
    retrieval_response: Dict[str, object],
) -> List[Dict[str, str]]:
    memories: List[str] = []
    for index, match in enumerate(results[:3], start=1):
        memories.append(
            "\n".join(
                [
                    f"Memory {index}",
                    f"Text: {match.record.text}",
                    f"Category: {match.record.category}",
                    f"Source: {match.record.source}",
                    f"Similarity: {match.score:.3f}",
                ]
            )
        )

    rejection_reason = str(retrieval_response.get("rejection_reason") or "none")
    confidence_score = retrieval_response.get("confidence_score")
    confidence_text = "n/a" if confidence_score is None else str(confidence_score)
    found_match = bool(retrieval_response.get("found"))

    system_prompt = (
        "You answer questions for Aila, a memory-based learning aid. "
        "Only use the retrieved memories provided in the user message. "
        "Do not use outside knowledge. "
        "If found_match is true, answer directly from Memory 1 in a short natural sentence. "
        "Do not mention confidence scores, retrieval summaries, similarity scores, or uncertainty when found_match is true. "
        "If the memory is about the user, answer in second person, like 'Your favorite food is pizza.' "
        "If found_match is false, or the memories are weak, conflicting, ambiguous, or missing, say you do not know yet based on stored memory. "
        "Keep the answer concise and natural."
    )
    user_prompt = "\n\n".join(
        [
            f"User question:\n{question}",
            (
                "Retrieval summary:\n"
                f"- found_match: {found_match}\n"
                f"- rejection_reason: {rejection_reason}\n"
                f"- confidence_score: {confidence_text}"
            ),
            "Retrieved memories:\n" + ("\n\n".join(memories) if memories else "None"),
        ]
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _looks_like_meta_answer(answer: str) -> bool:
    lowered = answer.casefold()
    if any(marker in lowered for marker in META_ANSWER_MARKERS):
        return True
    if lowered.startswith("i'm not sure") or lowered.startswith("i am not sure"):
        return True
    if re.search(r"\bconfidence\b|\bsimilarity\b", lowered):
        return True
    return False


def generate_grounded_answer(
    question: str,
    results: List[RetrievedMemory],
    retrieval_response: Dict[str, object],
    config: LocalLLMConfig,
) -> LocalLLMResult:
    if not config.enabled:
        return LocalLLMResult(used=False, answer=None, model=config.normalized_model)

    if not results:
        return LocalLLMResult(used=False, answer=None, model=config.normalized_model)

    if not retrieval_response.get("found"):
        return LocalLLMResult(used=False, answer=None, model=config.normalized_model)

    payload = {
        "model": config.normalized_model,
        "messages": _build_grounding_messages(question, results, retrieval_response),
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 80,
        },
    }

    try:
        response_payload = _http_post_json(
            f"{config.normalized_base_url}/api/chat",
            payload,
            timeout_seconds=config.timeout_seconds,
        )
    except Exception as exc:
        return LocalLLMResult(
            used=False,
            answer=None,
            model=config.normalized_model,
            error=_extract_error_message(exc),
        )

    message = response_payload.get("message", {})
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")

    cleaned_answer = content.strip()
    if not cleaned_answer:
        return LocalLLMResult(
            used=False,
            answer=None,
            model=config.normalized_model,
            error="The local model returned an empty answer.",
        )

    if _looks_like_meta_answer(cleaned_answer):
        return LocalLLMResult(
            used=False,
            answer=None,
            model=config.normalized_model,
            error="The local model returned a meta answer instead of a grounded one.",
        )

    usage = {
        "prompt_eval_count": response_payload.get("prompt_eval_count"),
        "eval_count": response_payload.get("eval_count"),
        "total_duration": response_payload.get("total_duration"),
    }

    return LocalLLMResult(
        used=True,
        answer=cleaned_answer,
        model=str(response_payload.get("model") or config.normalized_model),
        usage=usage,
    )
