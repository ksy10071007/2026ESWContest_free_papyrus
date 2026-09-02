"""Build the grounded system prompt used by the browser health chat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHAT_ROLE_PATH = PROJECT_DIR / "config" / "llm_chat_role.txt"
SCREENING_CONFIG_PATH = PROJECT_DIR / "config" / "screening_modalities.json"
MAX_CONTEXT_CHARACTERS = 12_000


class ChatPromptError(RuntimeError):
    """Raised when the local role or capability configuration is unusable."""


def load_chat_role_prompt(path: Path = CHAT_ROLE_PATH) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ChatPromptError("LLM chat role prompt is unavailable") from exc
    if not prompt:
        raise ChatPromptError("LLM chat role prompt is empty")
    return prompt


def load_service_capabilities(path: Path = SCREENING_CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChatPromptError("screening capability configuration is unavailable") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("modalities"), list):
        raise ChatPromptError("screening capability configuration is invalid")

    modalities = []
    for raw_item in payload["modalities"]:
        if not isinstance(raw_item, dict):
            continue
        modality_id = str(raw_item.get("id") or "").strip()
        if not modality_id:
            continue

        model_status = str(raw_item.get("model_status") or "not_configured").strip()
        item = {
            "id": modality_id,
            "label": str(raw_item.get("label") or modality_id),
            "short_description": str(raw_item.get("short_description") or ""),
            "camera_role": str(raw_item.get("camera_role") or ""),
            "model_status": model_status,
        }
        if model_status == "ready":
            classes = raw_item.get("classes")
            if isinstance(classes, list):
                item["classes"] = [str(value) for value in classes]
        modalities.append(item)

    return {
        "service_name": str(payload.get("service_name") or "MediFlow Kiosk"),
        "version": payload.get("version"),
        "modalities": modalities,
    }


def _bounded_json(value: Any, max_characters: int = MAX_CONTEXT_CHARACTERS) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    serialized = (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    if len(serialized) <= max_characters:
        return serialized
    return json.dumps(
        {
            "context_truncated": True,
            "original_characters": len(serialized),
            "preview": serialized[:max_characters],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_chat_system_prompt(
    diagnosis_result: Any,
    *,
    role_path: Path = CHAT_ROLE_PATH,
    screening_config_path: Path = SCREENING_CONFIG_PATH,
) -> str:
    role_prompt = load_chat_role_prompt(role_path)
    capabilities = load_service_capabilities(screening_config_path)
    return (
        f"{role_prompt}\n\n"
        "<service_capabilities>\n"
        f"{_bounded_json(capabilities)}\n"
        "</service_capabilities>\n\n"
        "<screening_result>\n"
        f"{_bounded_json(diagnosis_result)}\n"
        "</screening_result>"
    )
