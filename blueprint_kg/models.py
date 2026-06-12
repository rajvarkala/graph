"""Pydantic data models for all pipeline entities.

Every entity includes project_id for multi-project hierarchy.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Project(BaseModel):
    id: str
    name: str
    project_number: str = ""
    address: str = ""
    architect: str = ""
    mep_engineer: str | None = None
    structural_engineer: str | None = None
    owner: str = ""
    status: str = ""
    date: str = ""
    floor: str = ""
    total_area_sqft: float | None = None
    total_occupant_load: int | None = None
    source_pdfs: list[str] = Field(default_factory=list)


class Sheet(BaseModel):
    project_id: str
    number: str
    title: str = ""
    discipline: str = ""
    page_index: int = 0
    source_pdf: str = ""


class Space(BaseModel):
    project_id: str
    room_number: str
    name: str = ""
    floor: str = ""
    finish_floor: str | None = None
    finish_ceiling: str | None = None
    finish_wall: str | None = None
    partition_type: str | None = None
    door_type: str | None = None
    equipment_codes: list[str] = Field(default_factory=list)
    fixture_tags: list[str] = Field(default_factory=list)


class PartitionType(BaseModel):
    project_id: str
    code: str
    description: str = ""
    stc: int | None = None
    fire_rating: str | None = None
    thickness: str | None = None
    insulation: str | None = None
    test_ref: str | None = None
    variants: dict[str, str] | None = None


class FinishCode(BaseModel):
    project_id: str
    code: str
    category: str = ""
    name: str = ""
    manufacturer: str | None = None
    product: str | None = None
    color: str | None = None
    location: str | None = None
    spec_section: str | None = None


class Door(BaseModel):
    project_id: str
    door_number: str
    width: str | None = None
    height: str | None = None
    type: str = ""
    material: str | None = None
    finish: str | None = None
    glass_type: str | None = None
    frame_material: str | None = None
    hardware_set: str | None = None


class Equipment(BaseModel):
    project_id: str
    code: str
    name: str = ""
    location: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    procured_by: str | None = None
    ada_compliant: bool | None = None


class SpecSection(BaseModel):
    project_id: str
    section_number: str
    title: str = ""
    pages: str | None = None


class Abbreviation(BaseModel):
    project_id: str
    short: str
    full: str


class Detail(BaseModel):
    project_id: str
    number: str
    sheet: str
    description: str | None = None


class FixtureTag(BaseModel):
    project_id: str
    code: str
    description: str = ""


class CrossReference(BaseModel):
    project_id: str
    source_entity: str
    source_id: str
    target_sheet: str | None = None
    target_detail: str | None = None
    reference_text: str = ""


class StageOutput(BaseModel):
    """Container for a single stage's output."""
    stage: str
    project_id: str
    timestamp: str = ""
    entities: dict[str, list] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True