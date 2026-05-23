"""
LLM Provider abstraction layer.

Unified interface for Ollama and llama.cpp backends, preserving:
  - Structured output / guided decoding (JSON schema enforcement)
  - Temperature and context size control
  - Consistent response format regardless of backend

Currently supported backends (set via LLM_BACKEND env var):
  - "ollama"   (default): uses /api/chat endpoint
  - "llamacpp":           uses /v1/chat/completions endpoint (OpenAI-compatible)

Add new backends by subclassing LLMProvider and registering in _build_provider().
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import requests


BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"


class LLMProvider(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None):
        self.base_url = (base_url or os.environ.get(
            "LLM_API_URL", "http://localhost:11434"
        )).rstrip("/")
        self.model_name = model_name or os.environ.get("LLM_MODEL", "qwen3.5:27b")

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        num_ctx: int = 16384,
        timeout: int = 120,
    ) -> tuple[str, float]:
        ...

    @staticmethod
    def _json_from_response(data: dict, path: list[str]) -> Any:
        """Traverse nested dict following a key path."""
        current = data
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx] if idx < len(current) else None
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current


class OllamaProvider(LLMProvider):
    """Ollama backend via /api/chat with 'format' for guided decoding."""

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        num_ctx: int = 16384,
        timeout: int = 120,
    ) -> tuple[str, float]:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
            "think": False,
            "stream": False,
        }
        if schema:
            body["format"] = schema

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            return content, time.perf_counter() - t0
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[Ollama HTTP Error] {e}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"[Ollama Parse Error] {e}")


class LLamaCppProvider(LLMProvider):
    """llama.cpp backend via /v1/chat/completions (OpenAI-compatible)."""

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout: int = 120,
    ) -> tuple[str, float]:
        url = f"{self.base_url}/v1/chat/completions"
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            "max_tokens": 2048,
        }
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "schema": schema,
            }

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content, time.perf_counter() - t0
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[llama.cpp HTTP Error] {e}")
        except (ValueError, KeyError, IndexError) as e:
            raise RuntimeError(f"[llama.cpp Parse Error] {e}")


PROVIDER_MAP = {
    BACKEND_OLLAMA: OllamaProvider,
    BACKEND_LLAMACPP: LLamaCppProvider,
}


def build_provider(
    backend: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> LLMProvider:
    backend = (backend or os.environ.get("LLM_BACKEND", BACKEND_OLLAMA)).lower()
    cls = PROVIDER_MAP.get(backend)
    if cls is None:
        available = list(PROVIDER_MAP.keys())
        raise ValueError(f"Unknown LLM backend '{backend}'. Available: {available}")
    return cls(base_url=base_url, model_name=model_name)


def build_schema_for_format(input_schema: dict | None = None) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "service_id": {"type": "string"},
                        "url": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "SQL"],
                        },
                        "input": input_schema or {"type": ["string", "object", "null"]},
                    },
                    "required": ["task_name", "service_id", "url", "operation", "input"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }
