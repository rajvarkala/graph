"""Ollama Cloud API client for text LLM calls."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from pydantic import BaseModel
from rich.console import Console

console = Console()

OLLAMA_API_URL = "https://ollama.com/api/chat"


class OllamaClient:
    def __init__(
        self,
        model: str = "glm-5.1",
        api_key: str = "",
        max_retries: int = 3,
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature

    def chat(self, prompt: str, system: str = "", response_model: type[BaseModel] | None = None) -> dict | None:
        """Send a chat completion request. Optionally parse into a Pydantic model."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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
                        "temperature": self.temperature,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")

                if response_model:
                    try:
                        json_str = content
                        if "```json" in json_str:
                            json_str = json_str.split("```json")[1].split("```")[0]
                        elif "```" in json_str:
                            json_str = json_str.split("```")[1].split("```")[0]
                        parsed = json.loads(json_str)
                        return response_model(**parsed).model_dump()
                    except (json.JSONDecodeError, Exception) as e:
                        if attempt < self.max_retries - 1:
                            console.print(f"[yellow]  Retrying ({attempt+1}/{self.max_retries}): {e}[/yellow]")
                            time.sleep(2 ** attempt)
                            continue
                        console.print(f"[red]  Failed to parse LLM output after {self.max_retries} attempts[/red]")
                        return None

                return {"content": content, "raw": data}

            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]  HTTP {e.response.status_code}, retrying...[/yellow]")
                    time.sleep(2 ** attempt)
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]  Request error: {e}, retrying...[/yellow]")
                    time.sleep(2 ** attempt)
                    continue
                raise

        return None