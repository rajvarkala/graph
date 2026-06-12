"""Equipment extractor — extract equipment schedules."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import EQUIPMENT_EXTRACTOR_PROMPT
from blueprint_kg.models import Equipment


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[Equipment]:
    page_classification = config.page_classification if config else {}
    schedule_pages = page_classification.get("equipment_schedule", page_classification.get("schedule_pages", []))
    if schedule_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in schedule_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:15])

    result = client.chat(
        prompt=EQUIPMENT_EXTRACTOR_PROMPT.format(text=combined[:10000]),
        system="Extract equipment data. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        equipment = json.loads(content)
        return [Equipment(project_id=project_id, **{k: v for k, v in e.items() if k in Equipment.model_fields}) for e in equipment if "code" in e]
    except (json.JSONDecodeError, TypeError):
        return []