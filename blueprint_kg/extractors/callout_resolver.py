"""Callout resolver — extract cross-sheet references."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import CALLOUT_RESOLVER_PROMPT
from blueprint_kg.models import CrossReference


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[CrossReference]:
    page_classification = config.page_classification if config else {}
    callout_pages = page_classification.get("callout", page_classification.get("floor_plan", []))
    if callout_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in callout_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts)

    result = client.chat(
        prompt=CALLOUT_RESOLVER_PROMPT.format(text=combined[:15000]),
        system="Extract cross-reference callouts. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        callouts = json.loads(content)
        return [CrossReference(project_id=project_id, **{k: v for k, v in c.items() if k in CrossReference.model_fields}) for c in callouts if "reference_text" in c]
    except (json.JSONDecodeError, TypeError):
        return []