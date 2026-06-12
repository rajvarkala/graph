"""Page classifier — auto-detect page types from PDF text using LLM.

Scans each page's text content and classifies it into categories:
- sheet_index, floor_plan, finish_plan, finish_schedule, partition_type,
  door_schedule, equipment_schedule, rcp, abbreviation, callout, spec_section

Also identifies which pages need VLM (visual parsing) because text extraction
misses data in images, color swatches, logos, or dense multi-column tables.

Outputs both page_classification (page type → page indices) and vlm_pages
(page index → type + prompt) for downstream extractors.
"""

from __future__ import annotations

import json
from rich.console import Console

from blueprint_kg.llm.ollama_client import OllamaClient
from blueprint_kg.llm.prompts import PAGE_CLASSIFIER_PROMPT

console = Console()

PAGE_TYPES = [
    "sheet_index",
    "floor_plan",
    "finish_plan",
    "finish_schedule",
    "partition_type",
    "door_schedule",
    "equipment_schedule",
    "rcp",
    "abbreviation",
    "callout",
    "spec_section",
]

VLM_CANDIDATE_TYPES = {
    "floor_plan",
    "finish_plan",
    "finish_schedule",
    "partition_type",
    "door_schedule",
    "equipment_schedule",
}

VLM_PROMPTS = {
    "floor_plan": 'Extract all room numbers and their associated room names from this floor plan image. For each room, also identify any visible finish codes (floor like CPT-1, ceiling like ACS-1, wall like PNT-1) and partition type tags (like A41, B4, D1). Return a JSON array: [{"room_number": "", "name": "", "finish_floor": "", "finish_ceiling": "", "finish_wall": "", "partition_type": ""}]. Only include data clearly visible in the image.',
    "finish_plan": 'For each room on this finish plan image, list the floor finish code, ceiling code, and wall finish code. Return a JSON array: [{"room_number": "", "finish_floor": "", "finish_ceiling": "", "finish_wall": ""}]. Only include data clearly visible in the image.',
    "finish_schedule": 'Extract all finish schedule entries from this table image. Return a JSON array: [{"code": "", "category": "", "name": "", "manufacturer": "", "product": "", "color": "", "location": "", "spec_section": ""}]. Only include data clearly visible in the image.',
    "partition_type": 'Extract all partition types from this schedule image. Return a JSON array: [{"code": "", "description": "", "stc": null, "fire_rating": "", "thickness": "", "insulation": "", "test_ref": ""}]. Only include data clearly visible in the image.',
    "door_schedule": 'Extract all door types from this schedule image. Return a JSON array: [{"door_number": "", "width": "", "height": "", "type": "", "material": "", "finish": "", "glass_type": "", "frame_material": "", "hardware_set": ""}]. Only include data clearly visible in the image.',
    "equipment_schedule": 'Extract all equipment entries from this schedule image. Return a JSON array: [{"code": "", "name": "", "location": "", "manufacturer": "", "model": "", "procured_by": "", "ada_compliant": false}]. Only include data clearly visible in the image.',
}

NEEDS_VLM_SIGNALS = [
    "image",
    "color swatch",
    "logo",
    "graphic",
    "drawing",
    "floor plan",
    "plan view",
    "multi-column",
    "merged cells",
    "table with images",
    "finish plan",
    "finish schedule",
    "partition type schedule",
    "door schedule",
    "equipment schedule",
]


def classify_pages(
    texts: list[str],
    project_id: str,
    client: OllamaClient,
    config=None,
    max_pages_per_call: int = 15,
) -> dict:
    """Classify pages by type and identify VLM candidates.

    Returns:
        {
            "page_classification": {type: [page_indices]},
            "vlm_pages": [{"page": idx, "type": str, "prompt": str}],
            "page_details": [{"page": idx, "types": [str], "needs_vlm": bool, "reason": str}]
        }
    """
    if config and config.page_classification and config.vlm_pages:
        console.print("[dim]  Using config-provided page classification (override)[/dim]")
        return {
            "page_classification": config.page_classification,
            "vlm_pages": config.vlm_pages,
            "page_details": [],
        }

    console.print("[blue]  Auto-classifying pages...[/blue]")
    classification = {pt: [] for pt in PAGE_TYPES}
    vlm_pages = []
    page_details = []

    batches = []
    for start in range(0, len(texts), max_pages_per_call):
        batch_end = min(start + max_pages_per_call, len(texts))
        batch_texts = []
        for i in range(start, batch_end):
            snippet = texts[i][:600].strip()
            if snippet:
                batch_texts.append(f"--- PAGE {i} ---\n{snippet}")
        if batch_texts:
            batches.append((start, "\n\n".join(batch_texts)))

    for batch_start, batch_text in batches:
        prompt = PAGE_CLASSIFIER_PROMPT.format(
            page_types=", ".join(PAGE_TYPES),
            vlm_signals=", ".join(NEEDS_VLM_SIGNALS),
            text=batch_text,
        )
        result = client.chat(
            prompt=prompt,
            system="You are a construction document page classifier. Return ONLY valid JSON.",
            response_model=None,
        )
        if not result or "content" not in result:
            continue

        content = result["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]

        try:
            pages = json.loads(content)
            if isinstance(pages, dict):
                pages = pages.get("pages", [pages])
        except (json.JSONDecodeError, TypeError):
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    pages = json.loads(content[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    console.print(f"[yellow]    Page classification parse failed for batch starting at page {batch_start}[/yellow]")
                    continue
            else:
                continue

        for page_info in pages:
            page_idx = page_info.get("page", page_info.get("page_index", -1))
            if isinstance(page_idx, str) and page_idx.lstrip("-").isdigit():
                page_idx = int(page_idx)
            if not isinstance(page_idx, int) or page_idx < 0 or page_idx >= len(texts):
                continue

            page_types = page_info.get("types", page_info.get("type", []))
            if isinstance(page_types, str):
                page_types = [page_types]

            needs_vlm = page_info.get("needs_vlm", False)
            vlm_reason = page_info.get("vlm_reason", "")

            for pt in page_types:
                pt_lower = pt.lower().replace(" ", "_").replace("-", "_")
                if pt_lower in classification:
                    classification[pt_lower].append(page_idx)

            if needs_vlm:
                for pt in page_types:
                    pt_lower = pt.lower().replace(" ", "_").replace("-", "_")
                    if pt_lower in VLM_PROMPTS:
                        vlm_pages.append({
                            "page": page_idx,
                            "type": pt_lower,
                            "prompt": VLM_PROMPTS[pt_lower],
                        })
                        break

            page_details.append({
                "page": page_idx,
                "types": page_types,
                "needs_vlm": needs_vlm,
                "reason": vlm_reason,
            })

    classification = {k: sorted(set(v)) for k, v in classification.items() if v}

    console.print(f"[green]  Classified {len(page_details)} pages[/green]")
    for pt, indices in classification.items():
        console.print(f"[green]    {pt}: {len(indices)} pages (indices {indices[:5]}{'...' if len(indices) > 5 else ''})[/green]")
    console.print(f"[green]    VLM candidates: {len(vlm_pages)} pages[/green]")

    return {
        "page_classification": classification,
        "vlm_pages": vlm_pages,
        "page_details": page_details,
    }