"""Neo4j batch loaders — load entities and relationships."""

from __future__ import annotations

from blueprint_kg.graph.neo4j_client import Neo4jClient


def load_all(client: Neo4jClient, entities: dict, project_id: str) -> dict:
    """Load all entities into Neo4j. Returns counts per label."""
    counts = {}

    counts["Project"] = _load_project(client, entities, project_id)
    counts["Discipline"] = _load_disciplines(client, entities)
    counts["Sheet"] = _load_sheets(client, entities, project_id)
    counts["Manufacturer"] = _load_manufacturers(client, entities)
    counts["SpecSection"] = _load_spec_sections(client, entities, project_id)
    counts["FinishCode"] = _load_finish_codes(client, entities, project_id)
    counts["PartitionType"] = _load_partition_types(client, entities, project_id)
    counts["Space"] = _load_spaces(client, entities, project_id)
    counts["Door"] = _load_doors(client, entities, project_id)
    counts["Equipment"] = _load_equipment(client, entities, project_id)
    counts["Abbreviation"] = _load_abbreviations(client, entities, project_id)
    counts["Detail"] = _load_details(client, entities, project_id)
    counts["FixtureTag"] = _load_fixture_tags(client, entities, project_id)

    _load_relationships(client, entities, project_id)

    return counts


def _load_project(client: Neo4jClient, entities: dict, project_id: str) -> int:
    projects = entities.get("project", [])
    if not projects:
        client.run_query_no_return(
            "MERGE (p:Project {id: $id}) SET p.name = $id",
            {"id": project_id},
        )
        return 1
    for p in projects:
        client.run_query_no_return(
            "MERGE (proj:Project {id: $id}) SET proj += $props",
            {"id": p.get("id", project_id), "props": {k: v for k, v in p.items() if k != "id" and v is not None}},
        )
    return len(projects)


def _load_disciplines(client: Neo4jClient, entities: dict) -> int:
    disciplines = set()
    for s in entities.get("sheets", []):
        if s.get("discipline"):
            disciplines.add(s["discipline"])
    for d in disciplines:
        client.run_query_no_return("MERGE (d:Discipline {name: $name})", {"name": d})
    return len(disciplines)


def _load_sheets(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for s in entities.get("sheets", []):
        client.run_query_no_return(
            """MERGE (s:Sheet {project_id: $pid, number: $num})
            SET s.title = $title, s.discipline = $disc, s.page_index = $pi, s.source_pdf = $sp
            WITH s MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SHEET]->(s)""",
            {"pid": project_id, "num": s.get("number", ""), "title": s.get("title", ""),
             "disc": s.get("discipline", ""), "pi": s.get("page_index", 0),
             "sp": s.get("source_pdf", "")},
        )
    return len(entities.get("sheets", []))


def _load_manufacturers(client: Neo4jClient, entities: dict) -> int:
    mfrs = set()
    for f in entities.get("finishes", []):
        if f.get("manufacturer"):
            mfrs.add(f["manufacturer"])
    for e in entities.get("equipment", []):
        if e.get("manufacturer"):
            mfrs.add(e["manufacturer"])
    for m in mfrs:
        client.run_query_no_return("MERGE (m:Manufacturer {name: $name})", {"name": m})
    return len(mfrs)


def _load_spec_sections(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for s in entities.get("specs", []):
        client.run_query_no_return(
            """MERGE (ss:SpecSection {project_id: $pid, section_number: $sn})
            SET ss.title = $title, ss.pages = $pages
            WITH ss MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SPEC_SECTION]->(ss)""",
            {"pid": project_id, "sn": s.get("section_number", ""),
             "title": s.get("title", ""), "pages": s.get("pages")},
        )
    return len(entities.get("specs", []))


def _load_finish_codes(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for f in entities.get("finishes", []):
        client.run_query_no_return(
            """MERGE (fc:FinishCode {project_id: $pid, code: $code})
            SET fc.category = $cat, fc.name = $name, fc.manufacturer = $mfr,
                fc.product = $prod, fc.color = $color, fc.location = $loc, fc.spec_section = $ss
            WITH fc
            OPTIONAL MATCH (ss:SpecSection {project_id: $pid, section_number: $ss})
            WHERE $ss IS NOT NULL
            FOREACH (_ IN CASE WHEN ss IS NOT NULL THEN [1] ELSE [] END |
                MERGE (fc)-[:SPECIFIED_BY]->(ss))
            WITH fc
            OPTIONAL MATCH (m:Manufacturer {name: $mfr})
            WHERE $mfr IS NOT NULL
            FOREACH (_ IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
                MERGE (fc)-[:MADE_BY]->(m))""",
            {"pid": project_id, "code": f.get("code", ""), "cat": f.get("category", ""),
             "name": f.get("name", ""), "mfr": f.get("manufacturer"), "prod": f.get("product"),
             "color": f.get("color"), "loc": f.get("location"), "ss": f.get("spec_section")},
        )
    return len(entities.get("finishes", []))


def _load_partition_types(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for p in entities.get("partitions", []):
        client.run_query_no_return(
            """MERGE (pt:PartitionType {project_id: $pid, code: $code})
            SET pt.description = $desc, pt.stc = $stc, pt.fire_rating = $fr,
                pt.thickness = $thick, pt.insulation = $insul, pt.test_ref = $tr""",
            {"pid": project_id, "code": p.get("code", ""), "desc": p.get("description", ""),
             "stc": p.get("stc"), "fr": p.get("fire_rating"), "thick": p.get("thickness"),
             "insul": p.get("insulation"), "tr": p.get("test_ref")},
        )
    return len(entities.get("partitions", []))


def _load_spaces(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for s in entities.get("rooms", []):
        client.run_query_no_return(
            """MERGE (sp:Space {project_id: $pid, room_number: $rn})
            SET sp.name = $name, sp.floor = $floor
            WITH sp MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SPACE]->(sp)""",
            {"pid": project_id, "rn": s.get("room_number", ""), "name": s.get("name", ""),
             "floor": s.get("floor", "")},
        )
        # Finish relationships
        if s.get("finish_floor"):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (fc:FinishCode {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_FINISH_FLOOR]->(fc)""",
                {"pid": project_id, "rn": s["room_number"], "code": s["finish_floor"]},
            )
        if s.get("finish_ceiling"):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (fc:FinishCode {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_FINISH_CEIL]->(fc)""",
                {"pid": project_id, "rn": s["room_number"], "code": s["finish_ceiling"]},
            )
        if s.get("finish_wall"):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (fc:FinishCode {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_FINISH_WALL]->(fc)""",
                {"pid": project_id, "rn": s["room_number"], "code": s["finish_wall"]},
            )
        if s.get("partition_type"):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (pt:PartitionType {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_PARTITION]->(pt)""",
                {"pid": project_id, "rn": s["room_number"], "code": s["partition_type"]},
            )
        if s.get("door_type"):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (d:Door {project_id: $pid, type: $dtype})
                MERGE (sp)-[:HAS_DOOR]->(d)""",
                {"pid": project_id, "rn": s["room_number"], "dtype": s["door_type"]},
            )
    return len(entities.get("rooms", []))


def _load_doors(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for d in entities.get("doors", []):
        client.run_query_no_return(
            """MERGE (dr:Door {project_id: $pid, door_number: $dn})
            SET dr.type = $dtype, dr.width = $w, dr.height = $h,
                dr.material = $mat, dr.finish = $fin, dr.glass_type = $gt,
                dr.frame_material = $fm, dr.hardware_set = $hs""",
            {"pid": project_id, "dn": d.get("door_number", ""), "dtype": d.get("type", ""),
             "w": d.get("width"), "h": d.get("height"), "mat": d.get("material"),
             "fin": d.get("finish"), "gt": d.get("glass_type"),
             "fm": d.get("frame_material"), "hs": d.get("hardware_set")},
        )
    return len(entities.get("doors", []))


def _load_equipment(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for e in entities.get("equipment", []):
        client.run_query_no_return(
            """MERGE (eq:Equipment {project_id: $pid, code: $code})
            SET eq.name = $name, eq.location = $loc, eq.manufacturer = $mfr,
                eq.model = $model, eq.procured_by = $pb, eq.ada_compliant = $ada""",
            {"pid": project_id, "code": e.get("code", ""), "name": e.get("name", ""),
             "loc": e.get("location"), "mfr": e.get("manufacturer"),
             "model": e.get("model"), "pb": e.get("procured_by"),
             "ada": e.get("ada_compliant")},
        )
    return len(entities.get("equipment", []))


def _load_abbreviations(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for a in entities.get("abbreviations", []):
        client.run_query_no_return(
            "MERGE (ab:Abbreviation {project_id: $pid, short: $short}) SET ab.full = $full",
            {"pid": project_id, "short": a.get("short", ""), "full": a.get("full", "")},
        )
    return len(entities.get("abbreviations", []))


def _load_details(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for d in entities.get("callouts", []):
        if d.get("target_sheet") and d.get("target_detail"):
            detail_id = f"{d['target_detail']}/{d['target_sheet']}"
            client.run_query_no_return(
                """MERGE (dt:Detail {project_id: $pid, id: $did})
                SET dt.number = $num, dt.sheet = $sheet, dt.description = $desc
                WITH dt MATCH (s:Sheet {project_id: $pid, number: $sheet})
                MERGE (s)-[:CONTAINS_DETAIL]->(dt)""",
                {"pid": project_id, "did": detail_id, "num": d.get("target_detail", ""),
                 "sheet": d.get("target_sheet", ""), "desc": d.get("reference_text", "")},
            )
    return len(entities.get("callouts", []))


def _load_fixture_tags(client: Neo4jClient, entities: dict, project_id: str) -> int:
    for f in entities.get("fixtures", []):
        client.run_query_no_return(
            "MERGE (ft:FixtureTag {project_id: $pid, code: $code}) SET ft.description = $desc",
            {"pid": project_id, "code": f.get("code", ""), "desc": f.get("description", "")},
        )
    return len(entities.get("fixtures", []))


def _load_relationships(client: Neo4jClient, entities: dict, project_id: str):
    """Load remaining relationships that weren't created during node creation."""
    # Sheet -> Discipline
    for s in entities.get("sheets", []):
        if s.get("discipline"):
            client.run_query_no_return(
                """MATCH (s:Sheet {project_id: $pid, number: $num})
                MATCH (d:Discipline {name: $disc})
                MERGE (s)-[:BELONGS_TO]->(d)""",
                {"pid": project_id, "num": s.get("number", ""), "disc": s["discipline"]},
            )

    # Space -> Equipment
    for s in entities.get("rooms", []):
        for eq_code in s.get("equipment_codes", []):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (eq:Equipment {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_EQUIPMENT]->(eq)""",
                {"pid": project_id, "rn": s["room_number"], "code": eq_code},
            )
        for ft_code in s.get("fixture_tags", []):
            client.run_query_no_return(
                """MATCH (sp:Space {project_id: $pid, room_number: $rn})
                MATCH (ft:FixtureTag {project_id: $pid, code: $code})
                MERGE (sp)-[:HAS_FIXTURE]->(ft)""",
                {"pid": project_id, "rn": s["room_number"], "code": ft_code},
            )