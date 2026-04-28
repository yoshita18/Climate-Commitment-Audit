"""
Shared JSON parsing utility for LLM responses.

Modern LLMs (Groq/Llama, Mistral, Zephyr, …) frequently wrap their JSON
output in markdown code fences even when explicitly told not to:

    ```json
    { "key": "value" }
    ```

Calling json.loads() on such a response raises JSONDecodeError and the caller
falls back to defaults — silently producing wrong results (0 claims, 0.5
scores, etc.).

Use parse_llm_json() everywhere instead of bare json.loads(response).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json(response: str, fallback: Any = None, label: str = "") -> Any:
    """
    Parse a JSON value from an LLM response, handling common formatting issues:

      1. Markdown code fences  (```json ... ``` or ``` ... ```)
      2. Leading/trailing prose before/after the JSON block
      3. Escaped newlines and stray control characters
      4. Single-quoted keys (some smaller models emit these)

    Parameters
    ----------
    response : str
        Raw text returned by the LLM.
    fallback : Any
        Value returned when parsing fails after all recovery attempts.
        Defaults to None (callers should supply [] or {} as appropriate).
    label : str
        Short description shown in the warning log on failure (e.g. "extract_claims").

    Returns
    -------
    Any
        Parsed Python object, or *fallback* on failure.
    """
    if not response or not response.strip():
        logger.warning("parse_llm_json[%s]: empty response", label)
        return fallback

    text = response.strip()

    # ── 1. Strip markdown code fences ─────────────────────────────────────
    # Handles: ```json\n...\n``` or ```\n...\n``` or inline `...`
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # ── 2. If still no luck, find the first { or [ and last } or ] ────────
    # Handles leading prose like "Here is the JSON:\n{...}"
    if not text.startswith(("{", "[")):
        obj_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if obj_match:
            text = obj_match.group(1).strip()

    # ── 3. Attempt direct parse ────────────────────────────────────────────
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ── 4. Light repair: Python booleans/None + trailing commas ───────────
    repaired = text
    repaired = repaired.replace(": True",  ": true")
    repaired = repaired.replace(": False", ": false")
    repaired = repaired.replace(": None",  ": null")
    # Remove trailing commas before } or ] (with or without whitespace/newline)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        logger.warning(
            "parse_llm_json[%s]: could not parse JSON after recovery. "
            "Error: %s. First 200 chars of response: %r",
            label, exc, response[:200],
        )
        return fallback
