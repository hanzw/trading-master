"""LLM response caching layer using the DB cache table."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from functools import wraps

from ..db import get_db
from ..config import get_config

logger = logging.getLogger(__name__)

_caching_enabled = True


def set_caching_enabled(enabled: bool) -> None:
    """Toggle caching on/off."""
    global _caching_enabled
    _caching_enabled = enabled
    logger.info("LLM caching %s", "enabled" if enabled else "disabled")


def _make_cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
    """Generate a deterministic cache key from the LLM call parameters."""
    payload = json.dumps(
        {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"llm:{digest}"


def cached_llm_call(func):
    """Decorator for LLMClient.chat() that caches responses in the DB cache table.

    Cache key = sha256(model + system_prompt + user_prompt)[:32] prefixed with 'llm:'.
    Uses config.analysis.cache_ttl_hours for TTL.
    """

    @wraps(func)
    def wrapper(self, system_prompt: str, user_prompt: str):
        if not _caching_enabled:
            return func(self, system_prompt, user_prompt)

        cache_key = _make_cache_key(self.model, system_prompt, user_prompt)
        db = get_db()

        # Try cache hit
        cached = db.cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache HIT for key %s", cache_key)
            return cached["text"], cached["tokens"], 0.0

        # Cache miss — call through
        logger.debug("Cache MISS for key %s", cache_key)
        text, tokens, cost = func(self, system_prompt, user_prompt)

        # Store in cache
        ttl = get_config().analysis.cache_ttl_hours
        db.cache_set(
            cache_key,
            {"text": text, "tokens": tokens, "cost": cost},
            ttl_hours=ttl,
        )

        return text, tokens, cost

    return wrapper


def cached_llm_acall(func):
    """Async decorator for LLMClient.achat() that caches responses in the DB cache table.

    Same caching logic as cached_llm_call but for async methods.
    """

    @wraps(func)
    async def wrapper(self, system_prompt: str, user_prompt: str):
        if not _caching_enabled:
            return await func(self, system_prompt, user_prompt)

        cache_key = _make_cache_key(self.model, system_prompt, user_prompt)
        db = get_db()

        # Try cache hit
        cached = db.cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache HIT for key %s", cache_key)
            return cached["text"], cached["tokens"], 0.0

        # Cache miss — call through
        logger.debug("Cache MISS for key %s", cache_key)
        text, tokens, cost = await func(self, system_prompt, user_prompt)

        # Store in cache
        ttl = get_config().analysis.cache_ttl_hours
        db.cache_set(
            cache_key,
            {"text": text, "tokens": tokens, "cost": cost},
            ttl_hours=ttl,
        )

        return text, tokens, cost

    return wrapper
