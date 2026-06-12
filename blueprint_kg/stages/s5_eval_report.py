"""Stage 5: Eval & report — compute quality metrics, diff against baseline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.graph.neo4j_client import Neo4jClient

console = Console()


def load_baseline(baseline_path: str | None) -> dict | None:
    if not baseline_path:
        return None
    p = Path(baseline_path)
    if not p.exists():
        console.print(f"[yellow]Baseline not found: {baseline_path}[/yellow]")
        return None
    with open(p) as f:
        return json.load(f)


def compute_recall(pipeline_entities: dict, baseline: dict) -> dict:
    """Compute recall: % of baseline entities found by pipeline."""
    metrics = {}
    for entity_type in ["spaces", "sheets", "partitions", "finishes", "doors",
                         "equipment", "specs", "abbreviations"]:
        baseline_key = {
            "spaces": "spaces", "sheets": "sheets", "partitions": "partition_types",
            "finishes": "finish_codes", "doors": "doors", "equipment": "equipment",
            "specs": "spec_sections", "abbreviations": "abbreviations",
        }.get(entity_type, entity_type)

        if baseline_key not in baseline:
            continue

        baseline_items = baseline[baseline_key]
        if isinstance(baseline_items, dict):
            baseline_ids = set(baseline_items.keys())
        elif isinstance(baseline_items, list):
            id_field = {
                "spaces": "room_number", "sheets": "number",
                "partition_types": "code", "finish_codes": "code",
                "doors": "door_number", "equipment": "code",
                "spec_sections": "section_number", "abbreviations": "short",
            }.get(entity_type, "id")
            baseline_ids = {item.get(id_field, str(i)) for i, item in enumerate(baseline_items)}
        else:
            continue

        pipeline_key = {
            "spaces": "rooms", "sheets": "sheets", "partitions": "partitions",
            "finishes": "finishes", "doors": "doors", "equipment": "equipment",
            "specs": "specs", "abbreviations": "abbreviations",
        }.get(entity_type, entity_type)

        pipeline_items = pipeline_entities.get(pipeline_key, [])
        id_field = {
            "spaces": "room_number", "sheets": "number",
            "partitions": "code", "finishes": "code",
            "doors": "door_number", "equipment": "code",
            "specs": "section_number", "abbreviations": "short",
        }.get(entity_type, "id")
        pipeline_ids = {item.get(id_field, str(i)) for i, item in enumerate(pipeline_items)}

        found = len(baseline_ids & pipeline_ids)
        total = len(baseline_ids) if baseline_ids else 1
        metrics[entity_type] = {
            "baseline_count": len(baseline_ids),
            "found_count": found,
            "recall_pct": round(found / total * 100, 1) if total else 0,
        }

    return metrics


def compute_field_coverage(entities: dict) -> dict:
    """Compute average field coverage per entity type."""
    coverage = {}
    for entity_type, items in entities.items():
        if not items or not isinstance(items, list):
            continue
        total_fields = len(items[0]) if items else 0
        if total_fields == 0:
            continue
        filled = sum(
            sum(1 for v in item.values() if v is not None and v != "" and v != [])
            for item in items
        )
        max_fields = total_fields * len(items)
        coverage[entity_type] = round(filled / max_fields * 100, 1) if max_fields else 0
    return coverage


def query_neo4j_counts(client: Neo4jClient) -> dict:
    """Get node and relationship counts from Neo4j."""
    node_counts = {}
    for label in ["Project", "Sheet", "Space", "PartitionType", "FinishCode",
                   "Door", "Equipment", "SpecSection", "Abbreviation", "Detail",
                   "Manufacturer", "Discipline", "FixtureTag"]:
        result = client.run_query(f"MATCH (n:{label}) RETURN count(n) AS cnt")
        node_counts[label] = result[0]["cnt"] if result else 0

    rel_counts = {}
    for rel_type in ["HAS_SHEET", "HAS_SPACE", "HAS_PARTITION", "HAS_FINISH_FLOOR",
                      "HAS_FINISH_CEIL", "HAS_FINISH_WALL", "HAS_DOOR", "HAS_EQUIPMENT",
                      "SPECIFIED_BY", "MADE_BY", "BELONGS_TO"]:
        result = client.run_query(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt")
        rel_counts[rel_type] = result[0]["cnt"] if result else 0

    return {"nodes": node_counts, "relationships": rel_counts}


def generate_report(metrics: dict, neo4j_counts: dict, field_coverage: dict) -> str:
    """Generate markdown eval report."""
    lines = [
        "# Blueprint KG — Evaluation Report",
        f"\nGenerated: {datetime.now().isoformat()}\n",
        "## Entity Recall (vs Baseline)\n",
        "| Entity Type | Baseline | Found | Recall |",
        "|---|---|---|---|",
    ]
    for etype, m in metrics.get("recall", {}).items():
        lines.append(f"| {etype} | {m['baseline_count']} | {m['found_count']} | {m['recall_pct']}% |")

    lines.append("\n## Field Coverage\n")
    lines.append("| Entity Type | Coverage |")
    lines.append("|---|---|")
    for etype, pct in field_coverage.items():
        lines.append(f"| {etype} | {pct}% |")

    lines.append("\n## Neo4j Node Counts\n")
    lines.append("| Label | Count |")
    lines.append("|---|---|")
    for label, cnt in neo4j_counts.get("nodes", {}).items():
        lines.append(f"| {label} | {cnt} |")

    lines.append("\n## Neo4j Relationship Counts\n")
    lines.append("| Type | Count |")
    lines.append("|---|---|")
    for rtype, cnt in neo4j_counts.get("relationships", {}).items():
        lines.append(f"| {rtype} | {cnt} |")

    return "\n".join(lines)


def run_eval(config: Config, baseline_path: str | None = None) -> dict:
    """Run Stage 5: compute eval metrics and generate report."""
    s3_path = config.validated_dir / "s3_linked.json"
    with open(s3_path) as f:
        s3 = json.load(f)

    entities = s3.get("entities", {})

    baseline = load_baseline(baseline_path)
    recall = compute_recall(entities, baseline) if baseline else {}
    field_coverage = compute_field_coverage(entities)

    neo4j_cfg = config.neo4j_settings
    client = Neo4jClient(uri=neo4j_cfg["uri"], user=neo4j_cfg["user"],
                         password=neo4j_cfg["password"])
    neo4j_counts = query_neo4j_counts(client)
    client.close()

    metrics = {"recall": recall, "field_coverage": field_coverage}
    report = generate_report(metrics, neo4j_counts, field_coverage)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / "s5_eval_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    console.print(f"[green]Report written:[/green] {report_path}")
    return metrics