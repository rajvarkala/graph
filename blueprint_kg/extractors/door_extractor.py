"""Door extractor — extract door schedules."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import DOOR_EXTRACTOR_PROMPT
from blueprint_kg.models import Door


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[Door]:
    page_classification = config.page_classification if config else {}
    door_pages = page_classification.get("door_schedule", page_classification.get("door_schedule_pages", []))
    if door_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in door_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:10])

    result = client.chat(
        prompt=DOOR_EXTRACTOR_PROMPT.format(text=combined[:10000]),
        system="Extract door data. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        doors = json.loads(content)
        return [Door(project_id=project_id, **{k: v for k, v in d.items() if k in Door.model_fields}) for d in doors if "door_number" in d]
    except (json.JSONDecodeError, TypeError):
        return []