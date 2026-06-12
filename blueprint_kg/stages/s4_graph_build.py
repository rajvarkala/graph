"""Stage 4: Neo4j graph build — load entities and relationships."""

from __future__ import annotations

import json
from datetime import datetime

from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.graph.neo4j_client import Neo4jClient
from blueprint_kg.graph.schema import create_schema
from blueprint_kg.graph.loaders import load_all

console = Console()


def load_s3(config: Config) -> dict:
    with open(config.validated_dir / "s3_linked.json") as f:
        return json.load(f)


def run_build(config: Config) -> dict:
    """Run Stage 4: load validated entities into Neo4j."""
    s3 = load_s3(config)
    project_id = config.project_id
    neo4j_cfg = config.neo4j_settings

    client = Neo4jClient(
        uri=neo4j_cfg["uri"],
        user=neo4j_cfg["user"],
        password=neo4j_cfg["password"],
    )

    console.print("[blue]Creating schema...[/blue]")
    create_schema(client)

    console.print("[blue]Loading entities...[/blue]")
    counts = load_all(client, s3["entities"], project_id)

    client.close()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "node_counts": counts,
        "total_nodes": sum(counts.values()),
    }

    output_path = config.extracted_dir / "s4_graph_build.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    for label, count in counts.items():
        console.print(f"[green]  {label}:[/green] {count}")
    console.print(f"[green]Total nodes:[/green] {sum(counts.values())}")
    console.print(f"[green]Output:[/green] {output_path}")
    return metadata