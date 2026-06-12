"""Click CLI entry point for blueprint-kg."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Blueprint KG — construction PDFs → Neo4j knowledge graph."""


@cli.command()
@click.option("--drawings", required=True, help="Path to construction drawings PDF")
@click.option("--specs", required=True, help="Path to specifications PDF")
@click.option("--project", required=True, help="Project ID (e.g., fphq)")
@click.option("--project-name", default="", help="Full project name")
@click.option("--project-number", default="", help="Project number")
def ingest(drawings, specs, project, project_name, project_number):
    """Stage 0: Ingest PDFs, render pages, create Project entity."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s0_ingest import run_ingest

    config = Config(project)
    config.ensure_dirs()
    result = run_ingest(config, drawings, specs, project_name, project_number)
    console.print(f"[green]Stage 0 complete[/green]: {result}")


@cli.command()
@click.option("--project", required=True, help="Project ID")
def extract(project):
    """Stage 1: Extract structured entities from text using LLM."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s1_text_extract import run_extract

    config = Config(project)
    result = run_extract(config)
    console.print(f"[green]Stage 1 complete[/green]: {result}")


@cli.command()
@click.option("--project", required=True, help="Project ID")
@click.option("--pages", default="", help="Comma-separated page types to process")
@click.option("--threshold", default=80, help="Text coverage %% below which VLM is used")
def supplement(project, pages, threshold):
    """Stage 2: Supplement text extraction with VLM on rendered pages."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s2_vlm_supplement import run_supplement

    config = Config(project)
    page_list = [p.strip() for p in pages.split(",") if p.strip()] if pages else None
    result = run_supplement(config, page_list, threshold)
    console.print(f"[green]Stage 2 complete[/green]: {result}")


@cli.command()
@click.option("--project", required=True, help="Project ID")
def link(project):
    """Stage 3: Resolve cross-references and validate consistency."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s3_link_validate import run_link

    config = Config(project)
    result = run_link(config)
    console.print(f"[green]Stage 3 complete[/green]: {result}")


@cli.command()
@click.option("--project", required=True, help="Project ID")
@click.option("--neo4j-uri", default=None, help="Neo4j URI (overrides config)")
def build(project, neo4j_uri):
    """Stage 4: Load entities and relationships into Neo4j."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s4_graph_build import run_build

    config = Config(project)
    if neo4j_uri:
        config.neo4j_settings["uri"] = neo4j_uri
    result = run_build(config)
    console.print(f"[green]Stage 4 complete[/green]: {result}")


@cli.command()
@click.option("--project", required=True, help="Project ID")
@click.option("--baseline", default=None, help="Path to baseline JSON for comparison")
def eval(project, baseline):
    """Stage 5: Compute quality metrics and generate report."""
    from blueprint_kg.config import Config
    from blueprint_kg.stages.s5_eval_report import run_eval

    config = Config(project)
    baseline_path = baseline or config.baseline_file
    result = run_eval(config, baseline_path)
    console.print(f"[green]Stage 5 complete[/green]: {result}")


@cli.command()
@click.option("--drawings", required=True, help="Path to construction drawings PDF")
@click.option("--specs", required=True, help="Path to specifications PDF")
@click.option("--project", required=True, help="Project ID")
@click.option("--neo4j-uri", default=None, help="Neo4j URI")
@click.option("--baseline", default=None, help="Path to baseline JSON")
def run(drawings, specs, project, neo4j_uri, baseline):
    """Run the full pipeline: stages 0-5."""
    from blueprint_kg.config import Config

    config = Config(project)
    config.ensure_dirs()
    if neo4j_uri:
        config.neo4j_settings["uri"] = neo4j_uri

    console.print("[bold blue]Stage 0: Ingest[/bold blue]")
    from blueprint_kg.stages.s0_ingest import run_ingest
    run_ingest(config, drawings, specs)

    console.print("[bold blue]Stage 1: Text Extract[/bold blue]")
    from blueprint_kg.stages.s1_text_extract import run_extract
    run_extract(config)

    console.print("[bold blue]Stage 2: VLM Supplement[/bold blue]")
    from blueprint_kg.stages.s2_vlm_supplement import run_supplement
    run_supplement(config)

    console.print("[bold blue]Stage 3: Link & Validate[/bold blue]")
    from blueprint_kg.stages.s3_link_validate import run_link
    run_link(config)

    console.print("[bold blue]Stage 4: Graph Build[/bold blue]")
    from blueprint_kg.stages.s4_graph_build import run_build
    run_build(config)

    console.print("[bold blue]Stage 5: Eval & Report[/bold blue]")
    from blueprint_kg.stages.s5_eval_report import run_eval
    baseline_path = baseline or config.baseline_file
    run_eval(config, baseline_path)

    console.print("[bold green]Pipeline complete![/bold green]")


@cli.command()
@click.option("--project", required=True, help="Project ID")
def status(project):
    """Show pipeline stage completion status."""
    from blueprint_kg.config import Config
    import json

    config = Config(project)
    stages = ["s0_ingest", "s1_text_extract", "s2_vlm_supplement",
              "s3_linked", "s4_graph_build", "s5_eval_report"]
    for stage in stages:
        path = config.extracted_dir / f"{stage}.json"
        if stage.startswith("s3"):
            path = config.validated_dir / f"{stage}.json"
        elif stage.startswith("s5"):
            path = config.reports_dir / f"{stage}.md"
        exists = path.exists()
        icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
        console.print(f"  {icon} {stage}")


if __name__ == "__main__":
    cli()