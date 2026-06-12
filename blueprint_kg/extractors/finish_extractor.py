"""Finish code extractor — extract finish schedules."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import FINISH_EXTRACTOR_PROMPT
from blueprint_kg.models import FinishCode


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[FinishCode]:
    page_classification = config.page_classification if config else {}
    schedule_pages = page_classification.get("finish_schedule", page_classification.get("schedule_pages", []))
    if schedule_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in schedule_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:10])

    result = client.chat(
        prompt=FINISH_EXTRACTOR_PROMPT.format(text=combined[:10000]),
        system="Extract finish codes. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        finishes = json.loads(content)
        return [FinishCode(project_id=project_id, **{k: v for k, v in f.items() if k in FinishCode.model_fields}) for f in finishes if "code" in f]
    except (json.JSONDecodeError, TypeError):
        return []