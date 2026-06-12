#!/usr/bin/env python3
"""Full pipeline: LLM text extraction + VLM visual parsing → Neo4j → interactive graph.

Every extractor uses Ollama Cloud API. VLM processes floor plan images.
"""

import base64
import json
import re
import time
from pathlib import Path
from datetime import datetime

import fitz
import httpx
from neo4j import GraphDatabase

PROJECT_DIR = Path(__file__).parent
DRAWINGS_PDF = PROJECT_DIR / "1837_Input - Construction Drawings.pdf"
SPECS_PDF = PROJECT_DIR / "FPHQ Specs.pdf"
OUTPUT_DIR = PROJECT_DIR / "data" / "extracted"
RENDERED_DIR = PROJECT_DIR / "data" / "rendered_pages"

OLLAMA_API_KEY = "a0c6d8fd1eef40fbb22d578b9f1c667a.tNq40kyFN4qQFrCgkV49RsbE"
OLLAMA_API_URL = "https://ollama.com/api/chat"
TEXT_MODEL = "gemma3:4b"
VLM_MODEL = "qwen3-vl:235b"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "blueprint123"
PROJECT_ID = "fphq"


def call_llm(prompt: str, system: str = "You are a construction document parser. Return ONLY valid JSON array, no markdown, no explanation.", retries: int = 3) -> list[dict]:
    """Call Ollama Cloud text LLM. Returns parsed JSON list."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = httpx.post(
                OLLAMA_API_URL,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"},
                json={"model": TEXT_MODEL, "messages": messages, "temperature": 0.1, "stream": False},
                timeout=180.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.time() - t0
            print(f"    LLM call ({elapsed:.1f}s): {len(content)} chars")

            return parse_json_response(content)
        except Exception as e:
            print(f"    LLM call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


def call_vlm(image_path: str, prompt: str, retries: int = 2) -> list[dict]:
    """Call Ollama Cloud VLM with image. Returns parsed JSON list."""
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": "You are a construction document parser. Extract structured data from the image. Return ONLY valid JSON array. Do not invent data that is not visible."},
        {"role": "user", "content": prompt, "images": [b64_image]},
    ]

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = httpx.post(
                OLLAMA_API_URL,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"},
                json={"model": VLM_MODEL, "messages": messages, "temperature": 0.1, "stream": False},
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.time() - t0
            print(f"    VLM call ({elapsed:.1f}s): {len(content)} chars")
            return parse_json_response(content)
        except Exception as e:
            print(f"    VLM call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
    return []


def parse_json_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, handling code blocks."""
    if not content:
        return []
    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return [result]
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start:end+1])
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                pass
        print(f"    JSON parse failed, first 200 chars: {text[:200]}")
        return []


def extract_text_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    texts = [page.get_text("text") for page in doc]
    doc.close()
    return texts


def render_page(pdf_path: str, page_idx: int, output_dir: Path, dpi: int = 150) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    png_name = f"drawings_page_{page_idx:03d}.png"
    png_path = output_dir / png_name
    pix.save(str(png_path))
    doc.close()
    return str(png_path)


def run_pipeline():
    print("=" * 60)
    print("  Blueprint KG Pipeline — FPHQ (LLM + VLM)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    # ===== STAGE 0: Extract text and render pages =====
    print("\n=== Stage 0: Ingest ===")
    drawings_texts = extract_text_pages(str(DRAWINGS_PDF))
    specs_texts = extract_text_pages(str(SPECS_PDF))
    print(f"  Drawings: {len(drawings_texts)} pages")
    print(f"  Specs: {len(specs_texts)} pages")

    # Render key pages for VLM
    print("  Rendering key pages for VLM...")
    vlm_pages = {
        "floor_plan": render_page(str(DRAWINGS_PDF), 10, RENDERED_DIR),   # partition plan
        "finish_plan": render_page(str(DRAWINGS_PDF), 14, RENDERED_DIR),  # finish plan
        "finish_schedule": render_page(str(DRAWINGS_PDF), 7, RENDERED_DIR),  # finish schedule
        "partition_types": render_page(str(DRAWINGS_PDF), 4, RENDERED_DIR),  # partition types
        "door_schedule": render_page(str(DRAWINGS_PDF), 5, RENDERED_DIR),  # door types
        "equipment_schedule": render_page(str(DRAWINGS_PDF), 9, RENDERED_DIR),  # equipment
    }
    for name, path in vlm_pages.items():
        print(f"    {name}: {path}")

    project = {
        "id": PROJECT_ID,
        "name": "Foulger-Pratt Headquarters",
        "project_number": "860424",
        "address": "12435 Park Potomac Ave., Potomac, MD",
        "architect": "Perkins+Will",
        "mep_engineer": "WB Engineers+Consultants",
        "structural_engineer": "Cagley & Associates",
        "owner": "The Foulger-Pratt Companies",
        "status": "Issued for Permit and Bid",
        "date": "2015-05-28",
        "floor": "Level 2",
    }

    entities = {}

    # ===== STAGE 1: LLM Text Extraction =====
    print("\n=== Stage 1: LLM Text Extraction ===")

    # --- 1a. Sheets ---
    print("\n  [1/9] Extracting sheets...")
    prompt = """Extract all sheet numbers and titles from this construction document index page text.
Look for patterns like "A00-02 Cover Sheet", "A05-02 Level 2 Floor Plan", "E-401 Power Plan".
Return a JSON array: [{"number": "A00-02", "title": "Cover Sheet", "discipline": "Architectural"}]
Discipline should be: Architectural, Electrical, Mechanical, Plumbing, Fire Protection, Structural, or Civil.

Text:
""" + "\n---PAGE---\n".join(drawings_texts[:5])
    raw = call_llm(prompt)
    entities["sheets"] = [{"project_id": PROJECT_ID, **s} for s in raw if "number" in s or "sheet_number" in s]
    print(f"    → {len(entities['sheets'])} sheets")

    # --- 1b. Rooms/Spaces ---
    print("\n  [2/9] Extracting rooms/spaces...")
    prompt = """Extract all room numbers and names from this floor plan text.
Look for patterns like "250 CAFE", "205 CONFERENCE", "221 OFFICE", "272 SERVER".
Also capture any finish codes you see: floor (CPT-1, WD-1), ceiling (ACS-1, ACS-2), wall (PNT-1, UWS-1), partition types (A41, B4, D1).
Return a JSON array: [{"room_number": "250", "name": "CAFE", "floor": "Level 2", "finish_floor": "WD-1", "finish_ceiling": "ACS-3", "finish_wall": "PNT-2", "partition_type": "B41"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[10:15])
    raw = call_llm(prompt)
    entities["rooms"] = [{"project_id": PROJECT_ID, **r} for r in raw if "room_number" in r or "number" in r]
    # Normalize room_number field
    for r in entities["rooms"]:
        if "number" in r and "room_number" not in r:
            r["room_number"] = r.pop("number")
        r.setdefault("room_number", "")
        r.setdefault("name", "")
    print(f"    → {len(entities['rooms'])} rooms")

    # --- 1c. Partition Types ---
    print("\n  [3/9] Extracting partition types...")
    prompt = """Extract all partition types from this partition type schedule text.
Look for entries with codes like A41, A31, B41, B4, D1, D11, D21, A24, D91.
Each should have: code, description, STC rating, fire rating, thickness, insulation details.
Return a JSON array: [{"code": "A41", "description": "1-HR rated wall to structure above", "stc": 50, "fire_rating": "1-HR", "thickness": "5-5/8\"", "insulation": "Acoustic insulation"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[4:6])
    raw = call_llm(prompt)
    entities["partitions"] = [{"project_id": PROJECT_ID, **p} for p in raw if "code" in p]
    print(f"    → {len(entities['partitions'])} partition types")

    # --- 1d. Finish Codes ---
    print("\n  [4/9] Extracting finish codes...")
    prompt = """Extract all finish codes from this finish schedule text.
Look for codes like CPT-1, CPT-2, ACS-1, ACS-2, PNT-1, WD-1, RB-1, UWS-1, PL-1, GL-1, MTL-1, SSM-1, AF-1.
Each should have: code, category (carpet_tile, ceiling_system, paint, wood, resilient_base, upholstered_wall, plastic_laminate, glazing, ornamental_metal, solid_surface, window_film), name/description, manufacturer, product, color, location, spec_section.
Return a JSON array: [{"code": "CPT-2", "category": "carpet_tile", "name": "Interface Cubicle Heeling", "manufacturer": "Interface", "product": "Cubicle Heeling", "color": "", "location": "Private Offices", "spec_section": "09 68 13"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[7:10])
    raw = call_llm(prompt)
    entities["finishes"] = [{"project_id": PROJECT_ID, **f} for f in raw if "code" in f]
    print(f"    → {len(entities['finishes'])} finish codes")

    # --- 1e. Doors ---
    print("\n  [5/9] Extracting doors...")
    prompt = """Extract all door types from this door schedule text.
Look for door type codes like AGP, AGS, HM, FLUSH, WBD with their descriptions.
Return a JSON array: [{"door_number": "AGP", "type": "All Glass Pivoted", "material": "Aluminum", "glass_type": "Tempered", "frame_material": "Aluminum"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[5:7])
    raw = call_llm(prompt)
    entities["doors"] = [{"project_id": PROJECT_ID, **d} for d in raw if "door_number" in d or "type" in d]
    for d in entities["doors"]:
        if "door_number" not in d and "type" in d:
            d["door_number"] = d.get("type", "")
    print(f"    → {len(entities['doors'])} doors")

    # --- 1f. Equipment ---
    print("\n  [6/9] Extracting equipment...")
    prompt = """Extract all equipment entries from this equipment schedule text.
Look for codes like E1.01, E1.02, E1.07, E2.01 etc with their details.
Return a JSON array: [{"code": "E1.07", "name": "Refrigerator/Freezer Full-Height", "location": "Cafe", "manufacturer": "GE Monogram", "model": "ZISS480NHSS", "procured_by": "GC", "ada_compliant": false}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[9:12])
    raw = call_llm(prompt)
    entities["equipment"] = [{"project_id": PROJECT_ID, **e} for e in raw if "code" in e]
    print(f"    → {len(entities['equipment'])} equipment items")

    # --- 1g. Spec Sections ---
    print("\n  [7/9] Extracting spec sections...")
    prompt = """Extract all CSI specification section numbers and titles from this text.
Look for patterns like "09 51 13 - Acoustical Panel Ceilings", "09 68 13 - Tile Carpeting", "09 91 23 - Interior Painting".
Return a JSON array: [{"section_number": "09 68 13", "title": "Tile Carpeting"}]

Text:
""" + "\n---PAGE---\n".join(specs_texts[:30])
    raw = call_llm(prompt)
    entities["specs"] = [{"project_id": PROJECT_ID, **s} for s in raw if "section_number" in s]
    print(f"    → {len(entities['specs'])} spec sections")

    # --- 1h. Abbreviations ---
    print("\n  [8/9] Extracting abbreviations...")
    prompt = """Extract all abbreviation definitions from this construction document text.
Look for patterns like "AFF - Above Finished Floor", "CLG - Ceiling", "TYP - Typical", "GWB - Gypsum Wall Board".
Return a JSON array: [{"short": "AFF", "full": "Above Finished Floor"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[:4])
    raw = call_llm(prompt)
    entities["abbreviations"] = [{"project_id": PROJECT_ID, **a} for a in raw if "short" in a and "full" in a]
    print(f"    → {len(entities['abbreviations'])} abbreviations")

    # --- 1i. Cross-references ---
    print("\n  [9/9] Extracting cross-references...")
    prompt = """Extract all cross-reference callouts from this construction document text.
Look for patterns like "See 3/A00-71", "See Detail 1/A10-20", "Per Specification 09 68 13", "Refer to A00-70".
Return a JSON array: [{"source_entity": "Space", "source_id": "", "target_sheet": "A00-71", "target_detail": "3", "reference_text": "See 3/A00-71"}]

Text:
""" + "\n---PAGE---\n".join(drawings_texts[10:15])
    raw = call_llm(prompt)
    entities["callouts"] = [{"project_id": PROJECT_ID, **c} for c in raw if "reference_text" in c or "target_sheet" in c]
    print(f"    → {len(entities['callouts'])} cross-references")

    # ===== STAGE 2: VLM Visual Supplement =====
    print("\n=== Stage 2: VLM Visual Supplement ===")

    # --- VLM: Room layout from floor plan ---
    print("\n  [VLM 1/3] Room layout from floor plan...")
    vlm_prompt = """Look at this construction floor plan image. Extract all room numbers and their associated room names.
For each room, also identify the finish codes visible (floor finish like CPT-1, ceiling like ACS-1, wall like PNT-1) and partition type tags (like A41, B4, D1).
Return a JSON array: [{"room_number": "250", "name": "CAFE", "finish_floor": "WD-1", "finish_ceiling": "ACS-3", "finish_wall": "PNT-2", "partition_type": "B41"}]
Only include data you can clearly see in the image. Do not invent data."""
    vlm_rooms = call_vlm(vlm_pages["floor_plan"], vlm_prompt)
    print(f"    VLM found {len(vlm_rooms)} rooms")
    # Merge VLM rooms: add rooms not already in LLM results
    existing_rooms = {r.get("room_number", "") for r in entities.get("rooms", [])}
    vlm_added = 0
    for vr in vlm_rooms:
        rn = vr.get("room_number", "")
        if rn and rn not in existing_rooms:
            vr["project_id"] = PROJECT_ID
            vr.setdefault("floor", "Level 2")
            entities.setdefault("rooms", []).append(vr)
            existing_rooms.add(rn)
            vlm_added += 1
    # Also update finish/partition data for existing rooms from VLM
    for vr in vlm_rooms:
        rn = vr.get("room_number", "")
        if rn:
            for r in entities.get("rooms", []):
                if r.get("room_number") == rn:
                    if vr.get("finish_floor") and not r.get("finish_floor"):
                        r["finish_floor"] = vr["finish_floor"]
                    if vr.get("finish_ceiling") and not r.get("finish_ceiling"):
                        r["finish_ceiling"] = vr["finish_ceiling"]
                    if vr.get("finish_wall") and not r.get("finish_wall"):
                        r["finish_wall"] = vr["finish_wall"]
                    if vr.get("partition_type") and not r.get("partition_type"):
                        r["partition_type"] = vr["partition_type"]
    print(f"    Added {vlm_added} new rooms from VLM, updated finishes for existing rooms")

    # --- VLM: Finish schedule visual parsing ---
    print("\n  [VLM 2/3] Finish schedule from image...")
    vlm_prompt = """Look at this finish schedule image from a construction document.
Extract all finish code entries with their details: code, category, name, manufacturer, product, color, location, spec_section.
Return a JSON array: [{"code": "CPT-2", "category": "carpet_tile", "name": "Interface Cubicle Heeling", "manufacturer": "Interface", "product": "Cubicle Heeling", "color": "", "location": "Private Offices", "spec_section": "09 68 13"}]
Only include data you can clearly see in the image."""
    vlm_finishes = call_vlm(vlm_pages["finish_schedule"], vlm_prompt)
    print(f"    VLM found {len(vlm_finishes)} finish codes")
    existing_codes = {f.get("code", "") for f in entities.get("finishes", [])}
    vlm_finish_added = 0
    for vf in vlm_finishes:
        vf["project_id"] = PROJECT_ID
        code = vf.get("code", "")
        if code and code not in existing_codes:
            entities.setdefault("finishes", []).append(vf)
            existing_codes.add(code)
            vlm_finish_added += 1
        elif code:
            # Update missing fields in existing
            for f in entities["finishes"]:
                if f.get("code") == code:
                    for key in ["manufacturer", "product", "color", "location", "spec_section"]:
                        if vf.get(key) and not f.get(key):
                            f[key] = vf[key]
    print(f"    Added {vlm_finish_added} new finish codes from VLM")

    # --- VLM: Partition types visual parsing ---
    print("\n  [VLM 3/3] Partition types from image...")
    vlm_prompt = """Look at this partition type schedule image from a construction document.
Extract all partition type entries with: code (like A41, B4, D1), description, STC rating, fire rating, thickness, insulation.
Return a JSON array: [{"code": "A41", "description": "1-HR rated wall to structure above", "stc": 50, "fire_rating": "1-HR", "thickness": "5-5/8\"", "insulation": "Acoustic insulation"}]
Only include data you can clearly see in the image."""
    vlm_partitions = call_vlm(vlm_pages["partition_types"], vlm_prompt)
    print(f"    VLM found {len(vlm_partitions)} partition types")
    existing_pcodes = {p.get("code", "") for p in entities.get("partitions", [])}
    vlm_part_added = 0
    for vp in vlm_partitions:
        vp["project_id"] = PROJECT_ID
        code = vp.get("code", "")
        if code and code not in existing_pcodes:
            entities.setdefault("partitions", []).append(vp)
            existing_pcodes.add(code)
            vlm_part_added += 1
    print(f"    Added {vlm_part_added} new partition types from VLM")

    # ===== STAGE 3: Link & Validate =====
    print("\n=== Stage 3: Link & Validate ===")
    finish_codes = {f["code"] for f in entities.get("finishes", []) if "code" in f}
    partition_codes = {p["code"] for p in entities.get("partitions", []) if "code" in p}

    room_floor_codes = {r["finish_floor"] for r in entities.get("rooms", []) if r.get("finish_floor")}
    room_ceil_codes = {r["finish_ceiling"] for r in entities.get("rooms", []) if r.get("finish_ceiling")}
    room_wall_codes = {r["finish_wall"] for r in entities.get("rooms", []) if r.get("finish_wall")}
    room_partition_codes = {r["partition_type"] for r in entities.get("rooms", []) if r.get("partition_type")}
    all_room_codes = room_floor_codes | room_ceil_codes | room_wall_codes

    print(f"  Finish codes in schedule: {len(finish_codes)}")
    print(f"  Finish codes on plans: {len(all_room_codes)}")
    print(f"  Missing from schedule: {all_room_codes - finish_codes}")
    print(f"  Partition codes defined: {len(partition_codes)}")
    print(f"  Partition codes on plans: {len(room_partition_codes)}")
    print(f"  Missing partition types: {room_partition_codes - partition_codes}")

    # Save entities
    total = sum(len(v) for v in entities.items())
    with open(OUTPUT_DIR / "entities_llm.json", "w") as f:
        json.dump({"project_id": PROJECT_ID, "timestamp": datetime.now().isoformat(), "entities": entities,
                    "stats": {k: len(v) for k, v in entities.items()}}, f, indent=2, default=str)
    print(f"\n  Saved {total} entities to {OUTPUT_DIR / 'entities_llm.json'}")

    # ===== STAGE 4: Neo4j Graph Build =====
    print("\n=== Stage 4: Neo4j Graph Build ===")
    build_neo4j(entities, project)

    # ===== STAGE 5: Export & Visualize =====
    print("\n=== Stage 5: Export & Visualize ===")
    graph_data = export_graph_json()
    viz_path = create_viz_html(graph_data)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"\n  Neo4j browser: http://localhost:7474")
    print(f"  Graph viz:     http://localhost:8080/graph_viz.html")
    print(f"\n  Entities extracted:")
    for k, v in entities.items():
        print(f"    {k}: {len(v)}")
    print(f"  Total: {sum(len(v) for v in entities.values())}")


def build_neo4j(entities: dict, project: dict):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) WHERE n.project_id = $pid OR n.id = $pid DETACH DELETE n", pid=PROJECT_ID)

        # Constraints
        for c in [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Space) REQUIRE (s.project_id, s.room_number) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sheet) REQUIRE (s.project_id, s.number) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PartitionType) REQUIRE (p.project_id, p.code) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:FinishCode) REQUIRE (f.project_id, f.code) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Door) REQUIRE (d.project_id, d.door_number) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Equipment) REQUIRE (e.project_id, e.code) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SpecSection) REQUIRE (s.project_id, s.section_number) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Abbreviation) REQUIRE (a.project_id, a.short) IS UNIQUE",
        ]:
            session.run(c)

        # Project
        session.run("""
            MERGE (p:Project {id: $id})
            SET p.name = $name, p.project_number = $pn, p.address = $addr,
                p.architect = $arch, p.owner = $owner, p.status = $status,
                p.date = $date, p.floor = $floor
        """, id=PROJECT_ID, name=project.get("name", ""), pn=project.get("project_number", ""),
             addr=project.get("address", ""), arch=project.get("architect", ""),
             owner=project.get("owner", ""), status=project.get("status", ""),
             date=project.get("date", ""), floor=project.get("floor", ""))
        print("  ✓ Project")

        # Sheets
        for s in entities.get("sheets", []):
            num = s.get("number", "")
            if not num:
                continue
            session.run("""
                MERGE (sh:Sheet {project_id: $pid, number: $num})
                SET sh.title = $title, sh.discipline = $disc
                WITH sh MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SHEET]->(sh)
            """, pid=PROJECT_ID, num=num, title=s.get("title", ""), disc=s.get("discipline", ""))
        print(f"  ✓ {len(entities.get('sheets', []))} sheets")

        # Spaces
        for r in entities.get("rooms", []):
            rn = r.get("room_number", "")
            if not rn:
                continue
            session.run("""
                MERGE (sp:Space {project_id: $pid, room_number: $rn})
                SET sp.name = $name, sp.floor = $floor
                WITH sp MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SPACE]->(sp)
            """, pid=PROJECT_ID, rn=rn, name=r.get("name", ""), floor=r.get("floor", "Level 2"))

            if r.get("finish_floor"):
                session.run("""
                    MATCH (sp:Space {project_id: $pid, room_number: $rn})
                    MERGE (fc:FinishCode {project_id: $pid, code: $code})
                    MERGE (sp)-[:HAS_FINISH_FLOOR]->(fc)
                """, pid=PROJECT_ID, rn=rn, code=r["finish_floor"])
            if r.get("finish_ceiling"):
                session.run("""
                    MATCH (sp:Space {project_id: $pid, room_number: $rn})
                    MERGE (fc:FinishCode {project_id: $pid, code: $code})
                    MERGE (sp)-[:HAS_FINISH_CEIL]->(fc)
                """, pid=PROJECT_ID, rn=rn, code=r["finish_ceiling"])
            if r.get("finish_wall"):
                session.run("""
                    MATCH (sp:Space {project_id: $pid, room_number: $rn})
                    MERGE (fc:FinishCode {project_id: $pid, code: $code})
                    MERGE (sp)-[:HAS_FINISH_WALL]->(fc)
                """, pid=PROJECT_ID, rn=rn, code=r["finish_wall"])
            if r.get("partition_type"):
                session.run("""
                    MATCH (sp:Space {project_id: $pid, room_number: $rn})
                    MERGE (pt:PartitionType {project_id: $pid, code: $code})
                    MERGE (sp)-[:HAS_PARTITION]->(pt)
                """, pid=PROJECT_ID, rn=rn, code=r["partition_type"])
        print(f"  ✓ {len(entities.get('rooms', []))} spaces")

        # Partition Types
        for p in entities.get("partitions", []):
            code = p.get("code", "")
            if not code:
                continue
            session.run("""
                MERGE (pt:PartitionType {project_id: $pid, code: $code})
                SET pt.description = $desc, pt.stc = $stc, pt.fire_rating = $fr,
                    pt.thickness = $thick, pt.insulation = $insul
            """, pid=PROJECT_ID, code=code, desc=p.get("description", ""),
                 stc=p.get("stc"), fr=p.get("fire_rating"), thick=p.get("thickness"),
                 insul=p.get("insulation"))
        print(f"  ✓ {len(entities.get('partitions', []))} partition types")

        # Finish Codes
        for f in entities.get("finishes", []):
            code = f.get("code", "")
            if not code:
                continue
            session.run("""
                MERGE (fc:FinishCode {project_id: $pid, code: $code})
                SET fc.category = $cat, fc.name = $name, fc.manufacturer = $mfr,
                    fc.product = $prod, fc.color = $color, fc.location = $loc, fc.spec_section = $ss
            """, pid=PROJECT_ID, code=code, cat=f.get("category", ""), name=f.get("name", ""),
                 mfr=f.get("manufacturer"), prod=f.get("product"),
                 color=f.get("color"), loc=f.get("location"), ss=f.get("spec_section"))
            if f.get("manufacturer") and f["manufacturer"] not in ("N/A", "", None):
                session.run("""
                    MERGE (m:Manufacturer {name: $name})
                    WITH m MATCH (fc:FinishCode {project_id: $pid, code: $code})
                    MERGE (fc)-[:MADE_BY]->(m)
                """, name=f["manufacturer"], pid=PROJECT_ID, code=code)
            if f.get("spec_section"):
                session.run("""
                    MERGE (ss:SpecSection {project_id: $pid, section_number: $sn})
                    WITH ss MATCH (fc:FinishCode {project_id: $pid, code: $code})
                    MERGE (fc)-[:SPECIFIED_BY]->(ss)
                """, pid=PROJECT_ID, sn=f["spec_section"], code=code)
        print(f"  ✓ {len(entities.get('finishes', []))} finish codes")

        # Doors
        for d in entities.get("doors", []):
            dn = d.get("door_number", "")
            if not dn:
                continue
            session.run("""
                MERGE (dr:Door {project_id: $pid, door_number: $dn})
                SET dr.type = $dtype, dr.width = $w, dr.height = $h,
                    dr.material = $mat, dr.finish = $fin, dr.glass_type = $gt,
                    dr.frame_material = $fm, dr.hardware_set = $hs
            """, pid=PROJECT_ID, dn=dn, dtype=d.get("type", ""),
                 w=d.get("width"), h=d.get("height"), mat=d.get("material"),
                 fin=d.get("finish"), gt=d.get("glass_type"),
                 fm=d.get("frame_material"), hs=d.get("hardware_set"))
        print(f"  ✓ {len(entities.get('doors', []))} doors")

        # Equipment
        for e in entities.get("equipment", []):
            code = e.get("code", "")
            if not code:
                continue
            session.run("""
                MERGE (eq:Equipment {project_id: $pid, code: $code})
                SET eq.name = $name, eq.location = $loc, eq.manufacturer = $mfr,
                    eq.model = $model, eq.procured_by = $pb, eq.ada_compliant = $ada
            """, pid=PROJECT_ID, code=code, name=e.get("name", ""),
                 loc=e.get("location"), mfr=e.get("manufacturer"),
                 model=e.get("model"), pb=e.get("procured_by"),
                 ada=e.get("ada_compliant"))
        print(f"  ✓ {len(entities.get('equipment', []))} equipment items")

        # Spec Sections
        for s in entities.get("specs", []):
            sn = s.get("section_number", "")
            if not sn:
                continue
            session.run("""
                MERGE (ss:SpecSection {project_id: $pid, section_number: $sn})
                SET ss.title = $title
                WITH ss MATCH (p:Project {id: $pid}) MERGE (p)-[:HAS_SPEC_SECTION]->(ss)
            """, pid=PROJECT_ID, sn=sn, title=s.get("title", ""))
        print(f"  ✓ {len(entities.get('specs', []))} spec sections")

        # Abbreviations
        for a in entities.get("abbreviations", []):
            short = a.get("short", "")
            if not short:
                continue
            session.run("""
                MERGE (ab:Abbreviation {project_id: $pid, short: $short})
                SET ab.full = $full
            """, pid=PROJECT_ID, short=short, full=a.get("full", ""))
        print(f"  ✓ {len(entities.get('abbreviations', []))} abbreviations")

    driver.close()
    print("  ✓ Neo4j graph built")


def export_graph_json() -> dict:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes = []
    links = []
    node_set = set()

    def add_node(nid, label, group, radius):
        if nid not in node_set:
            nodes.append({"id": nid, "label": label, "group": group, "radius": radius})
            node_set.add(nid)

    with driver.session() as session:
        for r in session.run("MATCH (p:Project) RETURN p.id AS id, p.name AS name, p.floor AS floor, p.architect AS arch, p.owner AS owner, p.status AS status"):
            add_node(f"project_{r['id']}", r["name"] or r["id"], "Project", 30)

        for r in session.run("MATCH (p:Project)-[:HAS_SPACE]->(s:Space) RETURN p.id AS pid, s.room_number AS rn, s.name AS name"):
            add_node(f"space_{r['rn']}", f"{r['rn']} {r['name'] or ''}".strip(), "Space", 12)
            links.append({"source": f"project_{r['pid']}", "target": f"space_{r['rn']}"})

        for r in session.run("MATCH (p:Project)-[:HAS_SHEET]->(s:Sheet) RETURN p.id AS pid, s.number AS num, s.title AS title"):
            add_node(f"sheet_{r['num']}", f"{r['num']}", "Sheet", 7)
            links.append({"source": f"project_{r['pid']}", "target": f"sheet_{r['num']}"})

        for r in session.run("MATCH (p:Project)-[:HAS_SPEC_SECTION]->(ss:SpecSection) RETURN p.id AS pid, ss.section_number AS sn, ss.title AS title"):
            add_node(f"spec_{r['sn']}", f"{r['sn']}", "SpecSection", 8)
            links.append({"source": f"project_{r['pid']}", "target": f"spec_{r['sn']}"})

        for r in session.run("MATCH (fc:FinishCode) RETURN fc.code AS code, fc.name AS name, fc.category AS cat, fc.manufacturer AS mfr"):
            add_node(f"finish_{r['code']}", f"{r['code']}", "FinishCode", 7)

        for r in session.run("MATCH (s:Space)-[:HAS_FINISH_FLOOR]->(fc:FinishCode) RETURN s.room_number AS rn, fc.code AS code"):
            links.append({"source": f"space_{r['rn']}", "target": f"finish_{r['code']}"})
        for r in session.run("MATCH (s:Space)-[:HAS_FINISH_CEIL]->(fc:FinishCode) RETURN s.room_number AS rn, fc.code AS code"):
            links.append({"source": f"space_{r['rn']}", "target": f"finish_{r['code']}"})
        for r in session.run("MATCH (s:Space)-[:HAS_FINISH_WALL]->(fc:FinishCode) RETURN s.room_number AS rn, fc.code AS code"):
            links.append({"source": f"space_{r['rn']}", "target": f"finish_{r['code']}"})

        for r in session.run("MATCH (pt:PartitionType) RETURN pt.code AS code, pt.description AS desc, pt.fire_rating AS fr, pt.stc AS stc"):
            add_node(f"partition_{r['code']}", f"{r['code']}", "PartitionType", 7)
        for r in session.run("MATCH (s:Space)-[:HAS_PARTITION]->(pt:PartitionType) RETURN s.room_number AS rn, pt.code AS code"):
            links.append({"source": f"space_{r['rn']}", "target": f"partition_{r['code']}"})

        for r in session.run("MATCH (d:Door) RETURN d.door_number AS dn, d.type AS dtype"):
            add_node(f"door_{r['dn']}", f"{r['dn']}", "Door", 6)
        for r in session.run("MATCH (s:Space)-[:HAS_DOOR]->(d:Door) RETURN s.room_number AS rn, d.door_number AS dn"):
            links.append({"source": f"space_{r['rn']}", "target": f"door_{r['dn']}"})

        for r in session.run("MATCH (e:Equipment) RETURN e.code AS code, e.name AS name"):
            add_node(f"equip_{r['code']}", f"{r['code']}", "Equipment", 6)

        for r in session.run("MATCH (m:Manufacturer) RETURN m.name AS name"):
            add_node(f"mfr_{r['name']}", r["name"], "Manufacturer", 8)
        for r in session.run("MATCH (fc:FinishCode)-[:MADE_BY]->(m:Manufacturer) RETURN fc.code AS code, m.name AS name"):
            links.append({"source": f"finish_{r['code']}", "target": f"mfr_{r['name']}"})
        for r in session.run("MATCH (fc:FinishCode)-[:SPECIFIED_BY]->(ss:SpecSection) RETURN fc.code AS code, ss.section_number AS sn"):
            links.append({"source": f"finish_{r['code']}", "target": f"spec_{r['sn']}"})

        for r in session.run("MATCH (a:Abbreviation) RETURN a.short AS short, a.full AS full"):
            add_node(f"abbr_{r['short']}", f"{r['short']}", "Abbreviation", 4)

    driver.close()
    print(f"  → {len(nodes)} nodes, {len(links)} links")
    return {"nodes": nodes, "links": links}


def create_viz_html(graph_data: dict) -> Path:
    GROUP_COLORS = {
        "Project": "#a0a0ff", "Space": "#ff6b6b", "FinishCode": "#ffd93d",
        "PartitionType": "#6bcb77", "Door": "#4d96ff", "Equipment": "#ff922b",
        "SpecSection": "#cc5de8", "Manufacturer": "#20c997", "Sheet": "#74c0fc",
        "Abbreviation": "#868e96",
    }
    GROUP_LABELS = {
        "Project": "Project", "Space": "Rooms", "FinishCode": "Finishes",
        "PartitionType": "Partitions", "Door": "Doors", "Equipment": "Equipment",
        "SpecSection": "Specs", "Manufacturer": "Mfrs", "Sheet": "Sheets",
        "Abbreviation": "Abbrs",
    }
    group_counts = {}
    for n in graph_data["nodes"]:
        group_counts[n["group"]] = group_counts.get(n["group"], 0) + 1

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FPHQ Knowledge Graph — LLM+VLM Extracted</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;font-family:'Inter',system-ui,sans-serif;overflow:hidden;color:#e0e0f0}
#controls{position:fixed;top:16px;left:16px;z-index:100;background:rgba(15,15,30,0.92);
  border-radius:12px;padding:16px;backdrop-filter:blur(8px);border:1px solid rgba(100,100,200,0.2);
  max-height:90vh;overflow-y:auto;width:260px}
#controls h1{font-size:16px;color:#a0a0ff;margin-bottom:2px}
#controls h2{font-size:11px;color:#7070a0;font-weight:400;margin-bottom:10px}
.stats{font-size:11px;color:#7070a0;margin-bottom:8px}
.search{width:100%;padding:6px 10px;border-radius:6px;border:1px solid rgba(100,100,200,0.3);
  background:rgba(20,20,40,0.8);color:#e0e0f0;font-size:12px;margin-bottom:10px}
.search:focus{outline:none;border-color:rgba(100,100,200,0.6)}
.legend{display:flex;flex-wrap:wrap;gap:4px}
.legend-item{display:flex;align-items:center;gap:4px;font-size:10px;cursor:pointer;
  padding:2px 6px;border-radius:4px;transition:all 0.2s;user-select:none}
.legend-item:hover{background:rgba(100,100,200,0.3)}
.legend-item.hidden{opacity:0.3;text-decoration:line-through}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
#tooltip{position:fixed;pointer-events:none;background:rgba(15,15,30,0.95);
  border:1px solid rgba(100,100,200,0.4);border-radius:8px;padding:10px 14px;
  color:#e0e0f0;font-size:12px;max-width:300px;z-index:200;display:none}
#tooltip .tt-type{color:#7070a0;font-size:10px;text-transform:uppercase;letter-spacing:1px}
#tooltip .tt-label{color:#a0a0ff;font-weight:600;font-size:14px;margin:2px 0 4px}
#tooltip .tt-props{font-size:11px;line-height:1.6}
#tooltip .tt-props span{color:#a0a0ff}
svg{width:100vw;height:100vh}
.badge{display:inline-block;background:rgba(100,100,200,0.2);border-radius:4px;padding:1px 6px;font-size:10px;margin-left:4px;color:#a0a0ff}
</style>
</head>
<body>
<div id="controls">
  <h1>FPHQ Knowledge Graph</h1>
  <h2>LLM + VLM Extracted <span class="badge">Live</span></h2>
  <div class="stats" id="stats"></div>
  <input type="text" class="search" id="search" placeholder="Search nodes..." />
  <div class="legend" id="legend"></div>
</div>
<div id="tooltip">
  <div class="tt-type" id="tt-type"></div>
  <div class="tt-label" id="tt-label"></div>
  <div class="tt-props" id="tt-props"></div>
</div>
<svg id="graph"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const GD = ''' + json.dumps(graph_data) + ''';
const COLORS = ''' + json.dumps(GROUP_COLORS) + ''';
const LABELS = ''' + json.dumps(GROUP_LABELS) + ''';
const COUNTS = ''' + json.dumps(group_counts) + ''';

const svg = d3.select("#graph");
const W = window.innerWidth, H = window.innerHeight;
svg.attr("viewBox", [0, 0, W, H]);
const g = svg.append("g");

const zoom = d3.zoom().scaleExtent([0.05, 10]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const nodeMap = {};
GD.nodes.forEach(n => nodeMap[n.id] = n);
const links = GD.links.filter(l => nodeMap[l.source] && nodeMap[l.target]);

document.getElementById("stats").textContent = `${GD.nodes.length} nodes · ${links.length} links · LLM+VLM extracted`;

const legend = d3.select("#legend");
const visible = new Set(Object.keys(COLORS));

Object.entries(COLORS).forEach(([grp, color]) => {
  const item = legend.append("div").attr("class","legend-item").style("color",color)
    .on("click", () => { if(visible.has(grp)){visible.delete(grp);item.classed("hidden",true)}
      else{visible.add(grp);item.classed("hidden",false)} refresh(); });
  item.append("span").attr("class","legend-dot").style("background",color);
  item.append("span").text(`${LABELS[grp]} (${COUNTS[grp]||0})`);
});

function refresh() {
  nodeEls.attr("display", d => visible.has(d.group)?null:"none");
  linkEls.attr("display", d => visible.has(d.source.group)&&visible.has(d.target.group)?null:"none");
  textEls.attr("display", d => visible.has(d.group)?null:"none");
}

const sim = d3.forceSimulation(GD.nodes)
  .force("link", d3.forceLink(links).id(d=>d.id).distance(50).strength(0.4))
  .force("charge", d3.forceManyBody().strength(-100).distanceMax(350))
  .force("center", d3.forceCenter(W/2, H/2))
  .force("collision", d3.forceCollide().radius(d=>d.radius+3));

const linkEls = g.append("g").selectAll("line").data(links).join("line")
  .attr("stroke","rgba(100,100,200,0.1)").attr("stroke-width",1);

const nodeG = g.append("g").selectAll("g").data(GD.nodes).join("g")
  .call(d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y})
    .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y})
    .on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));

const nodeEls = nodeG.append("circle")
  .attr("r", d=>d.radius||8)
  .attr("fill", d=>COLORS[d.group]||"#666")
  .attr("stroke", d=>d.group==="Project"?"#fff":"none")
  .attr("stroke-width", d=>d.group==="Project"?2:0);

const textEls = nodeG.append("text")
  .text(d=>d.radius>=10?d.label:"")
  .attr("text-anchor","middle").attr("dy",d=>d.radius+11)
  .attr("font-size",d=>d.group==="Project"?"12px":"9px")
  .attr("fill","#c0c0e0").style("pointer-events","none");

const tt=document.getElementById("tooltip");
const ttType=document.getElementById("tt-type");
const ttLabel=document.getElementById("tt-label");
const ttProps=document.getElementById("tt-props");

nodeG.on("mouseover",(e,d)=>{
  tt.style.display="block"; ttType.textContent=d.group; ttLabel.textContent=d.label;
  const conn=new Set();
  links.forEach(l=>{if(l.source.id===d.id)conn.add(l.target.id);if(l.target.id===d.id)conn.add(l.source.id)});
  nodeEls.attr("opacity",n=>n.id===d.id||conn.has(n.id)?1:0.12);
  linkEls.attr("stroke",l=>l.source.id===d.id||l.target.id===d.id?"rgba(100,200,255,0.5)":"rgba(100,100,200,0.03)")
    .attr("stroke-width",l=>l.source.id===d.id||l.target.id===d.id?2:1);
  textEls.attr("opacity",n=>n.id===d.id||conn.has(n.id)?1:0.08);
  let props="";
  Object.entries(d).forEach(([k,v])=>{
    if(k!=="id"&&k!=="label"&&k!=="group"&&k!=="radius"&&k!=="index"&&v!=null&&v!=="")
      props+=`<span>${k}:</span> ${v}<br/>`;
  });
  ttProps.innerHTML=props;
}).on("mousemove",e=>{tt.style.left=(e.pageX+15)+"px";tt.style.top=(e.pageY-10)+"px"})
.on("mouseout",()=>{
  tt.style.display="none";
  nodeEls.attr("opacity",1);
  linkEls.attr("stroke","rgba(100,100,200,0.1)").attr("stroke-width",1);
  textEls.attr("opacity",1);
});

document.getElementById("search").addEventListener("input",e=>{
  const q=e.target.value.toLowerCase();
  if(!q){nodeEls.attr("opacity",1);return;}
  nodeEls.attr("opacity",n=>n.label.toLowerCase().includes(q)?1:0.08);
});

sim.on("tick",()=>{
  linkEls.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  nodeG.attr("transform",d=>`translate(${d.x},${d.y})`);
});

setTimeout(()=>{
  const b=g.node().getBBox();
  if(b.width>0){
    const s=0.85/Math.max(b.width/W,b.height/H);
    const tx=W/2-s*(b.x+b.width/2), ty=H/2-s*(b.y+b.height/2);
    svg.transition().duration(800).call(zoom.transform,d3.zoomIdentity.translate(tx,ty).scale(s));
  }
},2500);
</script>
</body>
</html>'''

    viz_path = PROJECT_DIR / "data" / "graph_viz.html"
    with open(viz_path, "w") as f:
        f.write(html)
    return viz_path


if __name__ == "__main__":
    run_pipeline()