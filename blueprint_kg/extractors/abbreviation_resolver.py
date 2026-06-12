"""Abbreviation resolver — extract abbreviation definitions."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import ABBREVIATION_RESOLVER_PROMPT
from blueprint_kg.models import Abbreviation


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[Abbreviation]:
    page_classification = config.page_classification if config else {}
    abbr_pages = page_classification.get("abbreviation", [])
    if abbr_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in abbr_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:5])

    result = client.chat(
        prompt=ABBREVIATION_RESOLVER_PROMPT.format(text=combined[:8000]),
        system="Extract abbreviation definitions. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        abbrs = json.loads(content)
        return [Abbreviation(project_id=project_id, **a) for a in abbrs if "short" in a and "full" in a]
    except (json.JSONDecodeError, TypeError):
        return []