"""LLM abstraction layer supporting OpenAI, Anthropic, and Ollama."""

from __future__ import annotations

import asyncio
import logging
import os

from ..config import get_config
from .cache import cached_llm_call, cached_llm_acall

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (input / output)
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}


class LLMClient:
    """Unified LLM client supporting openai, anthropic, and ollama providers."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "openai":
            import openai
            return openai.OpenAI()

        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic()
            except ImportError:
                raise ImportError(
                    "anthropic package is required for provider='anthropic'. "
                    "Install with: pip install anthropic"
                )

        if self.provider == "ollama":
            import openai
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return openai.OpenAI(base_url=base_url + "/v1", api_key="ollama")

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @cached_llm_call
    def chat(self, system_prompt: str, user_prompt: str) -> tuple[str, int, float]:
        """Send a chat request and return (response_text, tokens_used, cost_usd)."""
        if self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_prompt)
        # openai and ollama both use OpenAI-compatible API
        return self._chat_openai(system_prompt, user_prompt)

    @cached_llm_acall
    async def achat(self, system_prompt: str, user_prompt: str) -> tuple[str, int, float]:
        """Async chat: offloads the blocking LLM call to a thread so asyncio.gather works."""
        if self.provider == "anthropic":
            return await asyncio.to_thread(self._chat_anthropic, system_prompt, user_prompt)
        # openai and ollama both use OpenAI-compatible API
        return await asyncio.to_thread(self._chat_openai, system_prompt, user_prompt)

    def _chat_openai(self, system_prompt: str, user_prompt: str) -> tuple[str, int, float]:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens = usage.total_tokens if usage else 0
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self._estimate_cost(input_tokens, output_tokens)
        logger.debug("LLM request  [%s/%s] system=%r", self.provider, self.model, system_prompt)
        logger.debug("LLM request  [%s/%s] user=%r", self.provider, self.model, user_prompt)
        logger.debug("LLM response [%s/%s] text=%r", self.provider, self.model, text)
        logger.debug(
            "LLM [%s/%s] tokens=%d cost=$%.6f",
            self.provider, self.model, tokens, cost,
        )
        return text, tokens, cost

    def _chat_anthropic(self, system_prompt: str, user_prompt: str) -> tuple[str, int, float]:
        response = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        tokens = input_tokens + output_tokens
        cost = self._estimate_cost(input_tokens, output_tokens)
        logger.debug("LLM request  [%s/%s] system=%r", self.provider, self.model, system_prompt)
        logger.debug("LLM request  [%s/%s] user=%r", self.provider, self.model, user_prompt)
        logger.debug("LLM response [%s/%s] text=%r", self.provider, self.model, text)
        logger.debug(
            "LLM [%s/%s] tokens=%d cost=$%.6f",
            self.provider, self.model, tokens, cost,
        )
        return text, tokens, cost

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.provider == "ollama":
            return 0.0
        # Look up cost table; try exact match then partial match
        rates = _COST_TABLE.get(self.model)
        if rates is None:
            for key, val in _COST_TABLE.items():
                if key in self.model or self.model in key:
                    rates = val
                    break
        if rates is None:
            rates = (1.0, 3.0)  # fallback estimate
        input_cost = (input_tokens / 1_000_000) * rates[0]
        output_cost = (output_tokens / 1_000_000) * rates[1]
        return input_cost + output_cost


def get_llm() -> LLMClient:
    """Factory: create an LLMClient from app config."""
    cfg = get_config().llm
    return LLMClient(
        provider=cfg.provider,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
