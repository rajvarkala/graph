"""LLM prompt templates for all extractors."""

SHEET_INDEX_PROMPT = """Extract all sheet numbers and titles from this construction document index.

Look for patterns like:
- "A00-02" -> sheet number, "Cover Sheet" -> title
- "E-401" -> sheet number, "Power Plan" -> title
- "M-001" -> sheet number, "Mechanical Schedule" -> title

Return a JSON array of objects with: number, title, discipline.
Discipline should be one of: Architectural, Electrical, Mechanical, Plumbing, Fire Protection, Structural, Civil, Landscape.

Text:
{text}"""

ROOM_EXTRACTOR_PROMPT = """Extract all room numbers and names from this floor plan text.

Look for patterns like:
- "250 CAFE" -> room_number: "250", name: "CAFE"
- "205 BREAK ROOM" -> room_number: "205", name: "BREAK ROOM"

Also look for finish codes associated with rooms:
- Floor finishes (CPT-2, VCT-1, etc.)
- Ceiling finishes (ACS-1, ACS-2, etc.)
- Wall finishes (PNT-1, WD-1, etc.)
- Partition types (A41, B4, etc.)

Return a JSON array of objects with: room_number, name, floor, finish_floor, finish_ceiling, finish_wall, partition_type, door_type.

Text:
{text}"""

PARTITION_EXTRACTOR_PROMPT = """Extract all partition types from this partition type schedule.

Look for entries with:
- Code (e.g., A41, B4, D1)
- Description/construction details
- STC rating (number)
- Fire rating (e.g., "1-HR rated", "2-HR rated")
- Thickness
- Insulation
- Test reference (e.g., UL DES U465)
- Variants (sub-types like A41.1, A41.2)

Return a JSON array of objects with: code, description, stc, fire_rating, thickness, insulation, test_ref, variants.

Text:
{text}"""

FINISH_EXTRACTOR_PROMPT = """Extract all finish codes from this finish schedule.

Look for entries with:
- Code (e.g., CPT-2, ACS-1, PNT-1)
- Category (carpet_tile, ceiling_system, paint, wood, resilient, etc.)
- Name/description
- Manufacturer
- Product
- Color
- Location (where applied)
- Spec section reference (e.g., "09 68 13")

Return a JSON array of objects with: code, category, name, manufacturer, product, color, location, spec_section.

Text:
{text}"""

DOOR_EXTRACTOR_PROMPT = """Extract all door entries from this door schedule.

Look for entries with:
- Door number (e.g., 205A, 250A)
- Width and height
- Type (e.g., AGP, SD, HM)
- Material
- Finish
- Glass type (if applicable)
- Frame material
- Hardware set reference

Return a JSON array of objects with: door_number, width, height, type, material, finish, glass_type, frame_material, hardware_set.

Text:
{text}"""

EQUIPMENT_EXTRACTOR_PROMPT = """Extract all equipment entries from this equipment schedule.

Look for entries with:
- Code (e.g., E1.07, E2.04)
- Name/description
- Location (room number)
- Manufacturer
- Model
- Procured by (GC or Tenant)
- ADA compliant (true/false)

Return a JSON array of objects with: code, name, location, manufacturer, model, procured_by, ada_compliant.

Text:
{text}"""

SPEC_EXTRACTOR_PROMPT = """Extract all CSI specification sections from this text.

Look for patterns like:
- "09 68 13 - Tile Carpeting" -> section_number: "09 68 13", title: "Tile Carpeting"
- "09 51 13 - Acoustical Panel Ceilings" -> section_number: "09 51 13", title: "Acoustical Panel Ceilations"

Also look for manufacturer and product data within each section.

Return a JSON array of objects with: section_number, title, pages.

Text:
{text}"""

ABBREVIATION_RESOLVER_PROMPT = """Extract all abbreviation definitions from this text.

Look for patterns like:
- "AFF - Above Finished Floor"
- "CLG - Ceiling"
- "TYP - Typical"

Return a JSON array of objects with: short (abbreviation), full (expanded term).

Text:
{text}"""

PAGE_CLASSIFIER_PROMPT = """Classify each construction document page by its content type and determine if it needs visual (VLM) processing.

Page types: {page_types}

A page needs VLM processing if it contains: {vlm_signals}
These are pages where the meaningful data is in images, color swatches, logos, floor plan drawings, or dense multi-column tables that text extraction cannot capture reliably.

For each page, return a JSON array of objects:
- "page": the page index number
- "types": array of page type strings from the list above (a page can be multiple types)
- "needs_vlm": true if visual processing would extract data that text extraction misses
- "vlm_reason": brief reason why VLM is needed (or empty string if not needed)

Be thorough — most schedule pages and all plan pages need VLM. Spec text pages generally do not.

Text:
{text}"""

CALLOUT_RESOLVER_PROMPT = """Extract all cross-reference callouts from this construction document text.

Look for patterns like:
- "See 3/A00-71" -> reference to detail 3 on sheet A00-71
- "See Detail 1/A10-20" -> reference to detail 1 on sheet A10-20
- "Per Specification 09 68 13" -> reference to CSI section

Return a JSON array of objects with: source_entity (type), source_id, target_sheet, target_detail, reference_text.

Text:
{text}"""