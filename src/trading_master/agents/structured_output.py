"""Shared JSON parsing and LLM retry utilities for all agents."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str, model: Type[T] | None = None) -> dict | T:
    """Extract JSON from LLM response text.

    Tries three strategies in order:
    1. Direct ``json.loads`` on the full text.
    2. Extract content from a markdown code block (```json ... ```).
    3. Regex-extract the first ``{ ... }`` object.

    If *model* is provided, validates the parsed dict against that Pydantic
    model and returns the model instance.  If *model* is ``None``, returns
    the raw ``dict``.  Returns an empty ``dict`` (or raises ``ValidationError``)
    when parsing fails entirely.
    """
    data: dict = {}

    if text is None:
        return data if model is None else data

    # Strategy 1: direct parse
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: markdown code block
    if not data:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                data = json.loads(m.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

    # Strategy 3: regex extraction of first JSON object
    if not data:
        m = _JSON_OBJECT_RE.search(text)
        if m:
            try:
                data = json.loads(m.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

    if model is not None and data:
        return model.model_validate(data)

    return data


def llm_call_with_retry(
    llm,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T] | None = None,
    max_retries: int = 2,
) -> tuple[dict | T, int, float]:
    """Call LLM with retry on parse failure.

    Returns ``(parsed_result, total_tokens, total_cost)``.
    On retry, appends error feedback to the user prompt so the LLM can
    self-correct.
    """
    total_tokens = 0
    total_cost = 0.0
    last_error: Exception | None = None
    current_prompt = user_prompt

    for attempt in range(1 + max_retries):
        response_text, tokens, cost = llm.chat(system_prompt, current_prompt)
        total_tokens += tokens
        total_cost += cost

        try:
            result = parse_json_response(response_text, model=response_model)
            if isinstance(result, dict) and not result:
                raise ValueError("Failed to extract JSON from LLM response")
            return result, total_tokens, total_cost
        except (ValueError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "LLM JSON parse failed (attempt %d/%d): %s",
                attempt + 1, 1 + max_retries, exc,
            )
            # Build corrective prompt for retry
            current_prompt = (
                f"{user_prompt}\n\n"
                f"[SYSTEM: Your previous response was not valid JSON. "
                f"Error: {exc}. Please respond with ONLY valid JSON.]"
            )

    # All retries exhausted
    raise ValueError(
        f"Failed to parse LLM response after {1 + max_retries} attempts: {last_error}"
    )


async def async_llm_call_with_retry(
    llm,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T] | None = None,
    max_retries: int = 2,
) -> tuple[dict | T, int, float]:
    """Async version of llm_call_with_retry — uses llm.achat() for true async.

    Returns ``(parsed_result, total_tokens, total_cost)``.
    """
    total_tokens = 0
    total_cost = 0.0
    last_error: Exception | None = None
    current_prompt = user_prompt

    for attempt in range(1 + max_retries):
        response_text, tokens, cost = await llm.achat(system_prompt, current_prompt)
        total_tokens += tokens
        total_cost += cost

        try:
            result = parse_json_response(response_text, model=response_model)
            if isinstance(result, dict) and not result:
                raise ValueError("Failed to extract JSON from LLM response")
            return result, total_tokens, total_cost
        except (ValueError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "LLM JSON parse failed (attempt %d/%d): %s",
                attempt + 1, 1 + max_retries, exc,
            )
            current_prompt = (
                f"{user_prompt}\n\n"
                f"[SYSTEM: Your previous response was not valid JSON. "
                f"Error: {exc}. Please respond with ONLY valid JSON.]"
            )

    # All retries exhausted
    raise ValueError(
        f"Failed to parse LLM response after {1 + max_retries} attempts: {last_error}"
    )
