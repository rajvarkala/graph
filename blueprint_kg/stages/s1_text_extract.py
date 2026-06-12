"""Stage 1: Text extraction + LLM structuring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import fitz
from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.models import StageOutput
from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.extractors.page_classifier import classify_pages

console = Console()


def load_s0(config: Config) -> dict:
    path = config.extracted_dir / "s0_ingest.json"
    with open(path) as f:
        return json.load(f)


def extract_page_texts(pdf_path: str) -> list[str]:
    """Extract raw text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    texts = [page.get_text("text") for page in doc]
    doc.close()
    return texts


def run_extract(config: Config) -> dict:
    """Run Stage 1: classify pages, then extract structured entities using LLM."""
    from blueprint_kg.extractors import (
        sheet_index, room_extractor, partition_extractor,
        finish_extractor, door_extractor, equipment_extractor,
        spec_extractor, abbreviation_resolver, callout_resolver,
    )

    s0 = load_s0(config)
    project_id = config.project_id

    drawings_path = config.drawings_pdf
    specs_path = config.specs_pdf

    console.print("[blue]Extracting text from drawings...[/blue]")
    drawings_texts = extract_page_texts(drawings_path)
    console.print("[blue]Extracting text from specs...[/blue]")
    specs_texts = extract_page_texts(specs_path)

    client = OllamaClient(
        model=config.llm_settings.get("text_model", "glm-5.1"),
        api_key=config.ollama_api_key,
        max_retries=config.llm_settings.get("max_retries", 3),
        temperature=config.llm_settings.get("temperature", 0.1),
    )

    console.print("[blue]Classifying page types...[/blue]")
    page_info = classify_pages(drawings_texts, project_id, client, config)
    page_classification = page_info["page_classification"]
    vlm_pages = page_info["vlm_pages"]

    page_info_path = config.extracted_dir / "page_classification.json"
    with open(page_info_path, "w") as f:
        json.dump(page_info, f, indent=2, default=str)
    console.print(f"[green]Page classification saved:[/green] {page_info_path}")

    class ClassifiedConfig:
        """Minimal config-like object carrying classified page data."""
        def __init__(self, base_config, page_classification, vlm_pages):
            self._base = base_config
            self._classification = page_classification
            self._vlm_pages = vlm_pages

        @property
        def page_classification(self):
            return self._classification

        @property
        def vlm_pages(self):
            return self._vlm_pages

        def __getattr__(self, name):
            return getattr(self._base, name)

    classified_config = ClassifiedConfig(config, page_classification, vlm_pages)

    entities = {}
    extractors = [
        ("sheets", sheet_index, drawings_texts),
        ("rooms", room_extractor, drawings_texts),
        ("partitions", partition_extractor, drawings_texts),
        ("finishes", finish_extractor, drawings_texts),
        ("doors", door_extractor, drawings_texts),
        ("equipment", equipment_extractor, drawings_texts),
        ("specs", spec_extractor, specs_texts),
        ("abbreviations", abbreviation_resolver, drawings_texts),
        ("callouts", callout_resolver, drawings_texts),
    ]

    for name, extractor, texts in extractors:
        console.print(f"[blue]Extracting {name}...[/blue]")
        result = extractor.extract(texts, project_id, client, classified_config)
        entities[name] = [e.model_dump() for e in result]
        console.print(f"[green]  {name}:[/green] {len(result)} entities")

    output = StageOutput(
        stage="s1_text_extract",
        project_id=project_id,
        timestamp=datetime.now().isoformat(),
        entities=entities,
        metadata={
            "drawings_pages": len(drawings_texts),
            "specs_pages": len(specs_texts),
            "page_classification": page_classification,
            "vlm_pages_count": len(vlm_pages),
        },
    )

    output_path = config.extracted_dir / "s1_text_extract.json"
    with open(output_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2, default=str)

    console.print(f"[green]Output:[/green] {output_path}")
    return {"entity_types": len(entities), "total_entities": sum(len(v) for v in entities.values())}