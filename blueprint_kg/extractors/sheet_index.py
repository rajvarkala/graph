"""Sheet index extractor — extract sheet numbers and titles from title blocks."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import SHEET_INDEX_PROMPT
from blueprint_kg.models import Sheet


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[Sheet]:
    page_classification = config.page_classification if config else {}
    index_pages = page_classification.get("sheet_index", [])
    if index_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in index_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:5])
    result = client.chat(
        prompt=SHEET_INDEX_PROMPT.format(text=combined[:8000]),
        system="Extract sheet numbers and titles. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        sheets = json.loads(content)
        return [Sheet(project_id=project_id, **{k: v for k, v in s.items() if k in Sheet.model_fields}) for s in sheets if "number" in s]
    except (json.JSONDecodeError, TypeError):
        return []