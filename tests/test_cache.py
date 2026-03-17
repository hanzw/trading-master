"""Tests for LLM response caching."""

import pytest

from trading_master.agents.cache import _make_cache_key, set_caching_enabled, cached_llm_call
from trading_master.db import Database


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Create a temporary database and patch get_db to return it."""
    db_path = tmp_path / "test_cache.db"
    test_db = Database(db_path)
    monkeypatch.setattr("trading_master.agents.cache.get_db", lambda: test_db)
    return test_db


@pytest.fixture(autouse=True)
def reset_caching():
    """Ensure caching is enabled before each test."""
    set_caching_enabled(True)
    yield
    set_caching_enabled(True)


def test_cache_key_deterministic():
    key1 = _make_cache_key("gpt-4o-mini", "You are a helpful assistant.", "Hello")
    key2 = _make_cache_key("gpt-4o-mini", "You are a helpful assistant.", "Hello")
    assert key1 == key2
    assert key1.startswith("llm:")
    assert len(key1) == 4 + 32  # "llm:" + 32 hex chars


def test_cache_key_differs_with_different_inputs():
    key1 = _make_cache_key("gpt-4o-mini", "system", "user1")
    key2 = _make_cache_key("gpt-4o-mini", "system", "user2")
    assert key1 != key2

    key3 = _make_cache_key("gpt-4o", "system", "user1")
    assert key1 != key3


def test_cache_hit_returns_cached_data(db, monkeypatch):
    """Decorated function should return cached data on second call."""
    call_count = 0

    class FakeLLM:
        model = "gpt-4o-mini"

        @cached_llm_call
        def chat(self, system_prompt, user_prompt):
            nonlocal call_count
            call_count += 1
            return "response text", 100, 0.005

    # Patch get_config for TTL
    class FakeAnalysis:
        cache_ttl_hours = 4

    class FakeConfig:
        analysis = FakeAnalysis()

    monkeypatch.setattr(
        "trading_master.agents.cache.get_config", lambda: FakeConfig()
    )

    llm = FakeLLM()

    # First call — cache miss
    text1, tokens1, cost1 = llm.chat("sys", "user")
    assert text1 == "response text"
    assert tokens1 == 100
    assert cost1 == 0.005
    assert call_count == 1

    # Second call — cache hit
    text2, tokens2, cost2 = llm.chat("sys", "user")
    assert text2 == "response text"
    assert tokens2 == 100
    assert cost2 == 0.0  # Cache hit should return cost=0
    assert call_count == 1  # Should NOT have called through again


def test_cache_miss_calls_through(db, monkeypatch):
    """On cache miss, the underlying function should be called."""
    class FakeLLM:
        model = "gpt-4o-mini"

        @cached_llm_call
        def chat(self, system_prompt, user_prompt):
            return f"response for {user_prompt}", 200, 0.01

    class FakeAnalysis:
        cache_ttl_hours = 4

    class FakeConfig:
        analysis = FakeAnalysis()

    monkeypatch.setattr(
        "trading_master.agents.cache.get_config", lambda: FakeConfig()
    )

    llm = FakeLLM()
    text, tokens, cost = llm.chat("sys", "hello")
    assert text == "response for hello"
    assert tokens == 200
    assert cost == 0.01


def test_caching_disabled_bypasses_cache(db, monkeypatch):
    """When caching is disabled, every call goes through to the LLM."""
    call_count = 0

    class FakeLLM:
        model = "gpt-4o-mini"

        @cached_llm_call
        def chat(self, system_prompt, user_prompt):
            nonlocal call_count
            call_count += 1
            return "fresh response", 150, 0.007

    class FakeAnalysis:
        cache_ttl_hours = 4

    class FakeConfig:
        analysis = FakeAnalysis()

    monkeypatch.setattr(
        "trading_master.agents.cache.get_config", lambda: FakeConfig()
    )

    set_caching_enabled(False)
    llm = FakeLLM()

    llm.chat("sys", "user")
    llm.chat("sys", "user")
    assert call_count == 2  # Both calls should go through
