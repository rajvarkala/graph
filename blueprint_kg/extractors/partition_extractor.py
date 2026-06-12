"""Partition type extractor — extract partition schedules."""

from __future__ import annotations

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import PARTITION_EXTRACTOR_PROMPT
from blueprint_kg.models import PartitionType


def extract(texts: list[str], project_id: str, client: OllamaClient, config=None) -> list[PartitionType]:
    page_classification = config.page_classification if config else {}
    partition_pages = page_classification.get("partition_type", page_classification.get("partition_type_pages", []))
    if partition_pages:
        combined = "\n\n---PAGE---\n\n".join(texts[i] for i in partition_pages if i < len(texts))
    else:
        combined = "\n\n---PAGE---\n\n".join(texts[:10])

    result = client.chat(
        prompt=PARTITION_EXTRACTOR_PROMPT.format(text=combined[:10000]),
        system="Extract partition types. Return ONLY valid JSON array.",
        response_model=None,
    )
    if not result or "content" not in result:
        return []

    import json
    try:
        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        partitions = json.loads(content)
        return [PartitionType(project_id=project_id, **{k: v for k, v in p.items() if k in PartitionType.model_fields}) for p in partitions if "code" in p]
    except (json.JSONDecodeError, TypeError):
        return []