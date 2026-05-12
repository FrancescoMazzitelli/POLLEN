"""
petals-proxy: Ollama-compatible API proxy for Petals inference server.

Translates Ollama /api/chat requests (with structured JSON format enforcement)
into OpenAI-compatible /v1/chat/completions calls that Petals inference server supports.
"""

import json
import logging
import os
import re
import time
from typing import Any

import jsonschema
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("petals-proxy")

PETALS_API_URL = os.environ.get("PETALS_API_URL", "http://petals-inference:31337")
PETALS_MODEL = os.environ.get("PETALS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MAX_RETRIES = int(os.environ.get("PROXY_MAX_RETRIES", "3"))
PROXY_TIMEOUT = int(os.environ.get("PROXY_TIMEOUT", "180"))

app = FastAPI(title="petals-proxy", version="1.0")


class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[OllamaMessage]
    format: dict | None = Field(default=None)
    options: dict[str, Any] = Field(default_factory=dict)
    stream: bool = Field(default=False)


class OllamaResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool = True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def _build_json_system_prompt(schema: dict) -> str:
    return (
        "You must respond with ONLY valid JSON matching the schema below. "
        "Do NOT include markdown, explanations, or any text outside the JSON object. "
        "Return exactly the JSON structure specified.\n\n"
        f"SCHEMA:\n{json.dumps(schema, indent=2)}"
    )


def _inject_format_prompt(messages: list[dict], schema: dict | None) -> list[dict]:
    if not schema:
        return messages
    sys_instruction = _build_json_system_prompt(schema)
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = msg["content"] + "\n\n" + sys_instruction
            return messages
    return [{"role": "system", "content": sys_instruction}] + messages


def _call_petals(payload: dict, timeout: int) -> dict:
    url = f"{PETALS_API_URL}/v1/chat/completions"
    log.info("Calling Petals API at %s with model=%s", url, payload.get("model"))
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_json(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        return text[bracket_start : bracket_end + 1]
    return None


def _validate_json_schema(instance: Any, schema: dict) -> str | None:
    try:
        jsonschema.validate(instance, schema)
        return None
    except jsonschema.ValidationError as e:
        return str(e)


def _retry_with_stricter_prompt(messages: list[dict], schema: dict,
                                temperature: float, max_tokens: int,
                                timeout: int) -> str:
    raw_response = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        petals_payload = {
            "model": PETALS_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            result = _call_petals(petals_payload, timeout)
            raw = result["choices"][0]["message"]["content"]
            raw_response = raw
        except Exception as e:
            last_error = str(e)
            log.warning("Petals call failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            time.sleep(1)
            continue

        json_str = _extract_json(raw)
        if not json_str:
            log.warning("No JSON found in response (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Your response did not contain valid JSON. "
                           "Return ONLY valid JSON matching the schema. No other text."
            })
            continue

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning("JSON parse failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"JSON parse error: {e}. Return ONLY valid JSON."
            })
            continue

        err = _validate_json_schema(parsed, schema)
        if err is None:
            return json_str

        log.warning("Schema validation failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, err)
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Schema validation error: {err}. Fix and return valid JSON only."
        })

    raise RuntimeError(
        f"Failed to produce valid JSON after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}. "
        f"Last raw: {(raw_response or '')[:500]}"
    )


@app.post("/api/chat")
def chat(req: OllamaChatRequest):
    model_override = os.environ.get("PETALS_MODEL", PETALS_MODEL)

    messages = [m.model_dump() for m in req.messages]
    schema = req.format
    options = req.options

    temperature = options.get("temperature", 0.0)
    num_predict = options.get("num_predict", options.get("num_ctx", 4096))
    timeout = options.get("timeout", PROXY_TIMEOUT)

    if schema:
        messages = _inject_format_prompt(messages, schema)
        try:
            content = _retry_with_stricter_prompt(
                messages, schema, temperature, num_predict, timeout
            )
        except RuntimeError as e:
            log.error("Structured output failed: %s", e)
            raise HTTPException(status_code=502, detail=str(e)) from e
    else:
        petals_payload = {
            "model": model_override,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict,
            "stream": False,
        }
        try:
            result = _call_petals(petals_payload, timeout)
            content = result["choices"][0]["message"]["content"]
        except Exception as e:
            log.error("Petals call failed: %s", e)
            raise HTTPException(status_code=502, detail=str(e)) from e

    return {
        "model": req.model,
        "created_at": _now(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }


@app.get("/api/tags")
def list_models():
    return {
        "models": [
            {
                "name": PETALS_MODEL,
                "modified_at": _now(),
                "size": 0,
            }
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "backend": PETALS_API_URL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
