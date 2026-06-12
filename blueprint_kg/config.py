"""Configuration loading: YAML project configs + .env secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_DIR = Path(os.getenv("BLUEPRINT_DATA_DIR", "data"))
DEFAULT_CONFIG_DIR = Path(os.getenv("BLUEPRINT_CONFIG_DIR", "configs"))


class Config:
    def __init__(self, project_id: str, config_path: Path | None = None):
        self.project_id = project_id
        if config_path is None:
            config_path = DEFAULT_CONFIG_DIR / f"{project_id}.yaml"
        self.config_path = config_path
        self._raw: dict[str, Any] = {}
        self._load()

    def _load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            self._raw = yaml.safe_load(f) or {}

    @property
    def project(self) -> dict[str, Any]:
        return self._raw.get("project", {})

    @property
    def drawings_pdf(self) -> str:
        return self._raw.get("drawings_pdf", "")

    @property
    def specs_pdf(self) -> str:
        return self._raw.get("specs_pdf", "")

    @property
    def page_classification(self) -> dict[str, list[int]]:
        return self._raw.get("page_classification", {})

    @property
    def vlm_pages(self) -> list[dict[str, Any]]:
        return self._raw.get("vlm_pages", [])

    @property
    def llm_settings(self) -> dict[str, Any]:
        return self._raw.get("llm", {})

    @property
    def neo4j_settings(self) -> dict[str, Any]:
        raw = self._raw.get("neo4j", {})
        raw.setdefault("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        raw.setdefault("user", os.getenv("NEO4J_USER", "neo4j"))
        raw.setdefault("password", os.getenv("NEO4J_PASSWORD", "blueprint123"))
        return raw

    @property
    def baseline_file(self) -> str | None:
        return self._raw.get("baseline_file")

    @property
    def data_dir(self) -> Path:
        return DEFAULT_DATA_DIR

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def validated_dir(self) -> Path:
        return self.data_dir / "validated"

    @property
    def rendered_dir(self) -> Path:
        return self.data_dir / "rendered_pages"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def ollama_api_key(self) -> str:
        return os.getenv("OLLAMA_API_KEY", "")

    def ensure_dirs(self):
        for d in [self.data_dir, self.extracted_dir, self.validated_dir,
                  self.rendered_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)