"""VLM client for vision-language model calls with image input."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

console = Console()

OLLAMA_API_URL = "https://ollama.com/api/chat"


class VLMClient:
    def __init__(
        self,
        model: str = "qwen3-vl:235b",
        api_key: str = "",
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path: str, prompt: str, project_id: str = "") -> dict:
        """Send an image to the VLM with a structured prompt. Returns parsed entities."""
        b64_image = self._encode_image(image_path)

        system = (
            "You are a construction document parser. Extract structured data from "
            "the provided image. Return ONLY valid JSON. Do not invent data that is "
            "not visible in the image. If uncertain, omit the field rather than guess."
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            },
        ]

        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    OLLAMA_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.1,
                        "stream": False,
                    },
                    timeout=180.0,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")

                try:
                    json_str = content
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    parsed = json.loads(json_str)
                    return {
                        "entities": parsed if isinstance(parsed, list) else [parsed],
                        "new_count": len(parsed) if isinstance(parsed, list) else 1,
                        "conflict_count": 0,
                    }
                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        console.print(f"[yellow]  VLM JSON parse error, retrying...[/yellow]")
                        time.sleep(2 ** attempt)
                        continue
                    return {"entities": [], "new_count": 0, "conflict_count": 0, "raw": content}

            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        return {"entities": [], "new_count": 0, "conflict_count": 0}