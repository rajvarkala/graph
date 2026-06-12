"""Spec section extractor — extract CSI specification sections."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import SPEC_EXTRACTOR_PROMPT
from blueprint_kg.models import SpecSection


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[SpecSection]:
    page_classification = config.page_classification if config else {}
    spec_pages = page_classification.get("spec_section", [])
    if spec_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in spec_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:20])

    result = client.chat(
        prompt=SPEC_EXTRACTOR_PROMPT.format(text=combined[:15000]),
        system="Extract CSI specification sections. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        specs = json.loads(content)
        return [SpecSection(project_id=project_id, **{k: v for k, v in s.items() if k in SpecSection.model_fields}) for s in specs if "section_number" in s]
    except (json.JSONDecodeError, TypeError):
        return []