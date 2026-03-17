"""Cost budget enforcement for analysis runs."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when cost budget for an analysis run is exceeded."""


@dataclass
class CostBudget:
    max_cost_usd: float = 5.0
    warn_cost_usd: float = 2.0
    max_tokens: int = 500_000
    accumulated_cost: float = 0.0
    accumulated_tokens: int = 0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, tokens: int, cost: float) -> None:
        """Record usage. Raises BudgetExceededError if limits hit."""
        with self._lock:
            self.accumulated_cost += cost
            self.accumulated_tokens += tokens
            self.call_count += 1

            if self.accumulated_cost >= self.warn_cost_usd:
                logger.warning(
                    "Budget warning: $%.4f of $%.2f spent (%d tokens, %d calls)",
                    self.accumulated_cost,
                    self.max_cost_usd,
                    self.accumulated_tokens,
                    self.call_count,
                )

            if self.accumulated_cost > self.max_cost_usd:
                raise BudgetExceededError(
                    f"Cost budget exceeded: ${self.accumulated_cost:.4f} > "
                    f"${self.max_cost_usd:.2f} limit"
                )
            if self.accumulated_tokens > self.max_tokens:
                raise BudgetExceededError(
                    f"Token budget exceeded: {self.accumulated_tokens:,} > "
                    f"{self.max_tokens:,} limit"
                )

    def remaining_budget(self) -> float:
        """Return remaining USD budget."""
        return max(0.0, self.max_cost_usd - self.accumulated_cost)

    def summary(self) -> dict:
        """Return {cost, tokens, calls, remaining}."""
        return {
            "cost": self.accumulated_cost,
            "tokens": self.accumulated_tokens,
            "calls": self.call_count,
            "remaining": self.remaining_budget(),
        }

    def estimate_run_cost(self, n_tickers: int, model: str) -> float:
        """Estimate total cost for analyzing n tickers.

        ~7 LLM calls per ticker, estimate ~1500 tokens per call.
        Uses approximate cost rates from the LLM cost table.
        """
        from .agents.llm import _COST_TABLE

        # Look up rate; use input rate as rough average
        rates = _COST_TABLE.get(model)
        if rates is None:
            for key, val in _COST_TABLE.items():
                if key in model or model in key:
                    rates = val
                    break
        if rates is None:
            rates = (1.0, 3.0)  # fallback

        calls_per_ticker = 7
        tokens_per_call = 1500
        total_tokens = n_tickers * calls_per_ticker * tokens_per_call
        # Assume roughly 60% input, 40% output tokens
        input_tokens = int(total_tokens * 0.6)
        output_tokens = int(total_tokens * 0.4)
        input_cost = (input_tokens / 1_000_000) * rates[0]
        output_cost = (output_tokens / 1_000_000) * rates[1]
        return input_cost + output_cost
