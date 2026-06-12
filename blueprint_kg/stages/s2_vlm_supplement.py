"""Stage 2: VLM supplement — fill gaps with visual parsing.

Uses auto-classified VLM pages from Stage 1 if config doesn't specify vlm_pages.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.models import StageOutput
from blueprint_kg.llm.vlm_client import VLMClient

console = Console()


def load_s1(config: Config) -> dict:
    path = config.extracted_dir / "s1_text_extract.json"
    with open(path) as f:
        return json.load(f)


def load_page_classification(config: Config) -> dict | None:
    path = config.extracted_dir / "page_classification.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def run_supplement(config: Config, pages: list[str] | None = None, threshold: int = 80) -> dict:
    """Run Stage 2: VLM visual parsing on pages with low text coverage or visual value."""
    s1 = load_s1(config)
    project_id = config.project_id

    vlm_config = config.vlm_pages
    if not vlm_config:
        page_info = load_page_classification(config)
        if page_info and page_info.get("vlm_pages"):
            vlm_config = page_info["vlm_pages"]
            console.print(f"[blue]Using auto-detected VLM pages: {len(vlm_config)} pages[/blue]")
        else:
            console.print("[yellow]No VLM pages configured or auto-detected[/yellow]")
            vlm_config = []

    if pages:
        vlm_config = [p for p in vlm_config if p.get("type") in pages]

    client = VLMClient(
        model=config.llm_settings.get("vlm_model", "qwen3-vl:235b"),
        api_key=config.ollama_api_key,
        max_retries=config.llm_settings.get("max_retries", 3),
    )

    entities = {}
    total_new = 0
    total_conflicts = 0

    for page_cfg in vlm_config:
        page_type = page_cfg.get("type", "unknown")
        page_idx = page_cfg.get("page", 0)
        prompt = page_cfg.get("prompt", "")

        prefix = "drawings" if page_idx < 61 else "specs"
        png_path = config.rendered_dir / f"{prefix}_page_{page_idx:03d}.png"
        if not png_path.exists():
            console.print(f"[yellow]  Skipping page {page_idx}: PNG not found[/yellow]")
            continue

        console.print(f"[blue]  VLM processing page {page_idx} ({page_type})...[/blue]")
        result = client.analyze_image(str(png_path), prompt, project_id)
        entities[page_type] = result.get("entities", [])
        total_new += result.get("new_count", 0)
        total_conflicts += result.get("conflict_count", 0)

    output = StageOutput(
        stage="s2_vlm_supplement",
        project_id=project_id,
        timestamp=datetime.now().isoformat(),
        entities=entities,
        metadata={
            "vlm_pages_processed": len(vlm_config),
            "new_entities": total_new,
            "conflicts": total_conflicts,
        },
    )

    output_path = config.extracted_dir / "s2_vlm_supplement.json"
    with open(output_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2, default=str)

    console.print(f"[green]VLM pages processed:[/green] {len(vlm_config)}")
    console.print(f"[green]New entities:[/green] {total_new}")
    console.print(f"[green]Conflicts:[/green] {total_conflicts}")
    console.print(f"[green]Output:[/green] {output_path}")
    return output.metadata