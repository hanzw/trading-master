"""Tests for LLM cost estimation."""

from __future__ import annotations

import pytest

from trading_master.agents.llm import LLMClient, _COST_TABLE


# We can't instantiate LLMClient without an API key, so test _estimate_cost
# directly by creating a minimal instance with mocked client.


class _FakeLLMClient(LLMClient):
    """Subclass that skips actual client construction."""

    def _build_client(self):
        return None  # no real API client


@pytest.fixture
def openai_client():
    return _FakeLLMClient(provider="openai", model="gpt-4o-mini")


@pytest.fixture
def anthropic_client():
    return _FakeLLMClient(provider="anthropic", model="claude-3-5-sonnet")


@pytest.fixture
def ollama_client():
    return _FakeLLMClient(provider="ollama", model="llama3")


# ── Cost estimation ────────────────────────────────────────────────


class TestEstimateCost:
    def test_known_model_cost(self, openai_client):
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = openai_client._estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.60, abs=0.001)

    def test_zero_tokens(self, openai_client):
        cost = openai_client._estimate_cost(0, 0)
        assert cost == 0.0

    def test_anthropic_cost(self, anthropic_client):
        # claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
        cost = anthropic_client._estimate_cost(100_000, 50_000)
        expected = (100_000 / 1_000_000) * 3.0 + (50_000 / 1_000_000) * 15.0
        assert cost == pytest.approx(expected, abs=0.001)

    def test_ollama_always_free(self, ollama_client):
        cost = ollama_client._estimate_cost(1_000_000, 1_000_000)
        assert cost == 0.0

    def test_unknown_model_fallback(self):
        client = _FakeLLMClient(provider="openai", model="future-model-xyz")
        cost = client._estimate_cost(1_000_000, 1_000_000)
        # Fallback: $1.0/1M input, $3.0/1M output
        assert cost == pytest.approx(1.0 + 3.0, abs=0.001)

    def test_partial_model_match(self):
        # "claude-sonnet" should partial-match "claude-sonnet-4-20250514"
        client = _FakeLLMClient(provider="openai", model="claude-sonnet-4-20250514")
        cost = client._estimate_cost(1_000_000, 1_000_000)
        # Should match claude-sonnet entry: $3.00 + $15.00
        assert cost == pytest.approx(3.0 + 15.0, abs=0.001)

    def test_small_token_count(self, openai_client):
        cost = openai_client._estimate_cost(100, 200)
        assert cost > 0
        assert cost < 0.001  # very small

    def test_cost_table_has_entries(self):
        assert len(_COST_TABLE) > 0
        for model, (input_rate, output_rate) in _COST_TABLE.items():
            assert input_rate >= 0
            assert output_rate >= 0
