"""Stage 3: Link & validate — cross-reference resolution."""

from __future__ import annotations

import json
from datetime import datetime

from rich.console import Console

from blueprint_kg.config import Config
from blueprint_kg.models import StageOutput

console = Console()


def load_s1(config: Config) -> dict:
    with open(config.extracted_dir / "s1_text_extract.json") as f:
        return json.load(f)


def load_s2(config: Config) -> dict:
    path = config.extracted_dir / "s2_vlm_supplement.json"
    if not path.exists():
        return {"entities": {}}
    with open(path) as f:
        return json.load(f)


def resolve_callouts(entities: dict) -> tuple[list, list]:
    """Resolve cross-references like '3/A00-71' → detail 3 on sheet A00-71."""
    resolved = []
    unresolved = []
    sheets = {s["number"]: s for s in entities.get("sheets", [])}
    callouts = entities.get("callouts", [])

    for ref in callouts:
        target_sheet = ref.get("target_sheet", "")
        if target_sheet in sheets:
            resolved.append(ref)
        else:
            unresolved.append(ref)

    return resolved, unresolved


def check_finish_coverage(entities: dict) -> dict:
    """Check that finish codes on plans have schedule entries."""
    schedule_codes = {f["code"] for f in entities.get("finishes", [])}
    plan_codes = set()
    for room in entities.get("rooms", []):
        for field in ["finish_floor", "finish_ceiling", "finish_wall"]:
            code = room.get(field)
            if code:
                plan_codes.add(code)

    missing_from_schedule = plan_codes - schedule_codes
    unused_in_schedule = schedule_codes - plan_codes

    return {
        "plan_codes_count": len(plan_codes),
        "schedule_codes_count": len(schedule_codes),
        "missing_from_schedule": list(missing_from_schedule),
        "unused_in_schedule": list(unused_in_schedule),
        "coverage_pct": len(schedule_codes & plan_codes) / len(plan_codes) * 100 if plan_codes else 0,
    }


def check_partition_coverage(entities: dict) -> dict:
    """Check that partition type codes on plans have type definitions."""
    type_defs = {p["code"] for p in entities.get("partitions", [])}
    plan_types = {r["partition_type"] for r in entities.get("rooms", []) if r.get("partition_type")}

    missing = plan_types - type_defs
    return {
        "plan_types_count": len(plan_types),
        "type_defs_count": len(type_defs),
        "missing_from_defs": list(missing),
        "coverage_pct": len(type_defs & plan_types) / len(plan_types) * 100 if plan_types else 0,
    }


def resolve_abbreviations(entities: dict) -> dict:
    """Expand abbreviations using abbreviation map."""
    abbr_map = {a["short"]: a["full"] for a in entities.get("abbreviations", [])}
    expanded = 0
    for room in entities.get("rooms", []):
        if room.get("name") and room["name"].isupper():
            for short, full in abbr_map.items():
                if short in room["name"]:
                    room["name"] = room["name"].replace(short, f"{short} ({full})")
                    expanded += 1
    return {"abbreviations_expanded": expanded, "abbreviation_map_size": len(abbr_map)}


def map_specs_to_finishes(entities: dict) -> list[dict]:
    """Map finish codes to their spec sections."""
    mappings = []
    for finish in entities.get("finishes", []):
        if finish.get("spec_section"):
            mappings.append({
                "finish_code": finish["code"],
                "spec_section": finish["spec_section"],
                "manufacturer": finish.get("manufacturer"),
            })
    return mappings


def run_link(config: Config) -> dict:
    """Run Stage 3: resolve cross-references, validate consistency."""
    s1 = load_s1(config)
    s2 = load_s2(config)
    project_id = config.project_id

    entities = s1.get("entities", {})
    # Merge VLM supplements
    for entity_type, vlm_entities in s2.get("entities", {}).items():
        if entity_type in entities:
            existing_ids = {e.get("code") or e.get("room_number") or e.get("id")
                           for e in entities[entity_type]}
            for ve in vlm_entities:
                ve_id = ve.get("code") or ve.get("room_number") or ve.get("id")
                if ve_id not in existing_ids:
                    entities[entity_type].append(ve)
        else:
            entities[entity_type] = vlm_entities

    console.print("[blue]Resolving callouts...[/blue]")
    resolved, unresolved = resolve_callouts(entities)

    console.print("[blue]Checking finish coverage...[/blue]")
    finish_cov = check_finish_coverage(entities)

    console.print("[blue]Checking partition coverage...[/blue]")
    partition_cov = check_partition_coverage(entities)

    console.print("[blue]Resolving abbreviations...[/blue]")
    abbr_result = resolve_abbreviations(entities)

    console.print("[blue]Mapping specs to finishes...[/blue]")
    spec_mappings = map_specs_to_finishes(entities)

    output = StageOutput(
        stage="s3_linked",
        project_id=project_id,
        timestamp=datetime.now().isoformat(),
        entities=entities,
        metadata={
            "resolved_callouts": len(resolved),
            "unresolved_callouts": len(unresolved),
            "finish_coverage": finish_cov,
            "partition_coverage": partition_cov,
            "abbreviation_resolution": abbr_result,
            "spec_mappings_count": len(spec_mappings),
        },
    )

    config.validated_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.validated_dir / "s3_linked.json"
    with open(output_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2, default=str)

    console.print(f"[green]Resolved callouts:[/green] {len(resolved)}/{len(resolved)+len(unresolved)}")
    console.print(f"[green]Finish coverage:[/green] {finish_cov['coverage_pct']:.1f}%")
    console.print(f"[green]Partition coverage:[/green] {partition_cov['coverage_pct']:.1f}%")
    console.print(f"[green]Output:[/green] {output_path}")
    return output.metadata