"""Room extractor — extract room numbers, names, and finish assignments."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import ROOM_EXTRACTOR_PROMPT
from blueprint_kg.models import Space


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[Space]:
    page_classification = config.page_classification if config else {}
    floor_pages = page_classification.get("floor_plan", [])
    if floor_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in floor_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts)
    result = client.chat(
        prompt=ROOM_EXTRACTOR_PROMPT.format(text=combined[:12000]),
        system="Extract room data. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        rooms = json.loads(content)
        spaces = []
        for r in rooms:
            if "room_number" in r:
                spaces.append(Space(project_id=project_id, **{k: v for k, v in r.items() if k in Space.model_fields}))
        return spaces
    except (json.JSONDecodeError, TypeError):
        return []