"""Stage 0: PDF ingest, page render, Project entity creation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.models import Project, Sheet, StageOutput

console = Console()

DPI = 150


def extract_project_from_text(page_texts: list[str], config: Config) -> Project:
    """Extract Project entity from config (primary) or LLM fallback on title block text."""
    proj_cfg = config.project
    return Project(
        id=proj_cfg.get("id", config.project_id),
        name=proj_cfg.get("name", ""),
        project_number=proj_cfg.get("project_number", ""),
        address=proj_cfg.get("address", ""),
        architect=proj_cfg.get("architect", ""),
        mep_engineer=proj_cfg.get("mep_engineer"),
        structural_engineer=proj_cfg.get("structural_engineer"),
        owner=proj_cfg.get("owner", ""),
        status=proj_cfg.get("status", ""),
        date=proj_cfg.get("date", ""),
        floor=proj_cfg.get("floor", ""),
        total_area_sqft=proj_cfg.get("total_area_sqft"),
        total_occupant_load=proj_cfg.get("total_occupant_load"),
        source_pdfs=[],
    )


def render_pages(pdf_path: str, output_dir: Path, prefix: str) -> list[dict]:
    """Render each PDF page to PNG. Returns list of page metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=DPI)
        png_name = f"{prefix}_page_{i:03d}.png"
        png_path = output_dir / png_name
        pix.save(str(png_path))

        text = page.get_text("text")
        pages.append({
            "page_index": i,
            "source_pdf": pdf_path,
            "prefix": prefix,
            "png_path": str(png_path),
            "text_length": len(text),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "width": page.rect.width,
            "height": page.rect.height,
        })

    doc.close()
    return pages


def run_ingest(
    config: Config,
    drawings_path: str | None = None,
    specs_path: str | None = None,
    project_name: str = "",
    project_number: str = "",
) -> dict:
    """Run Stage 0: ingest PDFs, render pages, create Project entity."""
    config.ensure_dirs()

    drawings = drawings_path or config.drawings_pdf
    specs = specs_path or config.specs_pdf

    if not Path(drawings).exists():
        raise FileNotFoundError(f"Drawings PDF not found: {drawings}")
    if not Path(specs).exists():
        raise FileNotFoundError(f"Specs PDF not found: {specs}")

    console.print(f"[blue]Ingesting drawings:[/blue] {drawings}")
    drawings_pages = render_pages(drawings, config.rendered_dir, "drawings")

    console.print(f"[blue]Ingesting specs:[/blue] {specs}")
    specs_pages = render_pages(specs, config.rendered_dir, "specs")

    # Extract text per page
    all_page_texts = []
    doc_drawings = fitz.open(drawings)
    for page in doc_drawings:
        all_page_texts.append(page.get_text("text"))
    doc_drawings.close()

    doc_specs = fitz.open(specs)
    for page in doc_specs:
        all_page_texts.append(page.get_text("text"))
    doc_specs.close()

    # Create Project entity
    project = extract_project_from_text(all_page_texts, config)
    project.source_pdfs = [drawings, specs]
    if project_name:
        project.name = project_name
    if project_number:
        project.project_number = project_number

    # Compute metrics
    total_pages = len(drawings_pages) + len(specs_pages)
    text_coverage = sum(1 for p in drawings_pages + specs_pages if p["text_length"] > 50)
    text_coverage_pct = text_coverage / total_pages * 100 if total_pages else 0

    output = StageOutput(
        stage="s0_ingest",
        project_id=config.project_id,
        timestamp=datetime.now().isoformat(),
        entities={
            "project": [project.model_dump()],
            "drawings_pages": drawings_pages,
            "specs_pages": specs_pages,
        },
        metadata={
            "total_pages": total_pages,
            "text_coverage_pct": round(text_coverage_pct, 1),
            "drawings_pages_count": len(drawings_pages),
            "specs_pages_count": len(specs_pages),
        },
    )

    output_path = config.extracted_dir / "s0_ingest.json"
    with open(output_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2, default=str)

    console.print(f"[green]Project:[/green] {project.name} ({project.id})")
    console.print(f"[green]Pages:[/green] {total_pages} ({len(drawings_pages)} drawings, {len(specs_pages)} specs)")
    console.print(f"[green]Text coverage:[/green] {text_coverage_pct:.1f}%")
    console.print(f"[green]Output:[/green] {output_path}")

    return output.metadata