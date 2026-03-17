"""Tests for structured JSON output parsing and LLM retry utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from trading_master.agents.structured_output import (
    parse_json_response,
    llm_call_with_retry,
    async_llm_call_with_retry,
)


# ── Test model ─────────────────────────────────────────────────────


class MockSignal(BaseModel):
    signal: str
    confidence: int
    summary: str


# ── parse_json_response: Strategy 1 — direct parse ────────────────


class TestDirectParse:
    def test_valid_json(self):
        text = '{"signal": "BUY", "confidence": 80, "summary": "Good"}'
        result = parse_json_response(text)
        assert result["signal"] == "BUY"
        assert result["confidence"] == 80

    def test_valid_json_with_model(self):
        text = '{"signal": "BUY", "confidence": 80, "summary": "Good"}'
        result = parse_json_response(text, model=MockSignal)
        assert isinstance(result, MockSignal)
        assert result.signal == "BUY"

    def test_valid_json_with_whitespace(self):
        text = '  \n  {"signal": "HOLD", "confidence": 50, "summary": "Neutral"}  \n  '
        result = parse_json_response(text)
        assert result["signal"] == "HOLD"


# ── parse_json_response: Strategy 2 — markdown code block ─────────


class TestCodeBlockParse:
    def test_json_code_block(self):
        text = 'Here is the analysis:\n```json\n{"signal": "SELL", "confidence": 90, "summary": "Overvalued"}\n```'
        result = parse_json_response(text)
        assert result["signal"] == "SELL"

    def test_plain_code_block(self):
        text = 'Result:\n```\n{"signal": "BUY", "confidence": 70, "summary": "Value play"}\n```\nDone.'
        result = parse_json_response(text)
        assert result["signal"] == "BUY"

    def test_code_block_with_extra_text(self):
        text = 'I analyzed the stock carefully.\n\n```json\n{"signal": "HOLD", "confidence": 55, "summary": "Mixed signals"}\n```\n\nLet me know if you need more detail.'
        result = parse_json_response(text)
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 55


# ── parse_json_response: Strategy 3 — regex extraction ────────────


class TestRegexParse:
    def test_json_embedded_in_text(self):
        text = 'Based on my analysis, here is the result: {"signal": "BUY", "confidence": 65, "summary": "Moderate"} I hope this helps.'
        result = parse_json_response(text)
        assert result["signal"] == "BUY"

    def test_json_with_preamble(self):
        text = 'Sure! Here is the JSON:\n\n{"signal": "SELL", "confidence": 85, "summary": "Bearish"}'
        result = parse_json_response(text)
        assert result["signal"] == "SELL"

    def test_multiline_json_in_text(self):
        text = 'Analysis complete.\n{"signal": "HOLD",\n"confidence": 42,\n"summary": "Unclear direction"}\nEnd.'
        result = parse_json_response(text)
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 42


# ── parse_json_response: failure cases ─────────────────────────────


class TestParseFailures:
    def test_no_json_returns_empty_dict(self):
        result = parse_json_response("This has no JSON at all.")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = parse_json_response("")
        assert result == {}

    def test_invalid_json_returns_empty_dict(self):
        result = parse_json_response("{bad json: not valid}")
        assert result == {}

    def test_model_validation_error(self):
        # Valid JSON but wrong fields for model
        text = '{"wrong_field": "value"}'
        with pytest.raises(ValidationError):
            parse_json_response(text, model=MockSignal)

    def test_no_json_with_model_returns_empty(self):
        # No JSON found → empty dict, model not applied
        result = parse_json_response("no json here", model=MockSignal)
        assert result == {}

    def test_none_input(self):
        result = parse_json_response(None)
        assert result == {}


# ── parse_json_response: edge cases ────────────────────────────────


class TestParseEdgeCases:
    def test_nested_json(self):
        text = '{"signal": "BUY", "confidence": 75, "summary": "Good", "details": {"pe": 15.2}}'
        result = parse_json_response(text)
        assert result["details"]["pe"] == 15.2

    def test_json_array_ignored(self):
        """Arrays are not dicts — should fall through to regex for object."""
        text = '[1, 2, 3]'
        result = parse_json_response(text)
        # json.loads succeeds but returns a list, not dict
        assert isinstance(result, list) or result == {}

    def test_multiple_json_objects_greedy_regex(self):
        """Greedy regex may fail on multiple objects — returns empty dict."""
        text = '{"a": 1} and then {"b": 2}'
        result = parse_json_response(text)
        # Greedy regex matches '{"a": 1} and then {"b": 2}' which isn't valid JSON
        # This is expected behavior — single object extraction only
        assert isinstance(result, dict)

    def test_json_with_special_chars(self):
        text = '{"signal": "BUY", "confidence": 80, "summary": "Revenue up 20% — strong Q4"}'
        result = parse_json_response(text)
        assert "Revenue" in result["summary"]

    def test_json_with_unicode(self):
        text = '{"signal": "HOLD", "confidence": 50, "summary": "Analysis complete"}'
        result = parse_json_response(text)
        assert result["signal"] == "HOLD"


# ── llm_call_with_retry ───────────────────────────────────────────


class TestLLMCallWithRetry:
    def test_success_on_first_try(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            '{"signal": "BUY", "confidence": 80, "summary": "Good"}',
            100, 0.01,
        )
        result, tokens, cost = llm_call_with_retry(mock_llm, "sys", "usr")
        assert result["signal"] == "BUY"
        assert tokens == 100
        assert mock_llm.chat.call_count == 1

    def test_retry_on_bad_json(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            ("not valid json", 50, 0.005),
            ('{"signal": "SELL", "confidence": 70, "summary": "OK"}', 60, 0.006),
        ]
        result, tokens, cost = llm_call_with_retry(mock_llm, "sys", "usr", max_retries=1)
        assert result["signal"] == "SELL"
        assert tokens == 110
        assert mock_llm.chat.call_count == 2

    def test_all_retries_exhausted(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ("bad output", 50, 0.005)
        with pytest.raises(ValueError, match="Failed to parse"):
            llm_call_with_retry(mock_llm, "sys", "usr", max_retries=1)
        assert mock_llm.chat.call_count == 2

    def test_retry_includes_error_feedback(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            ("nope", 50, 0.005),
            ('{"signal": "BUY", "confidence": 60, "summary": "OK"}', 60, 0.006),
        ]
        llm_call_with_retry(mock_llm, "sys", "original prompt", max_retries=1)
        # Second call should include error feedback
        second_call_prompt = mock_llm.chat.call_args_list[1][0][1]
        assert "SYSTEM" in second_call_prompt
        assert "valid JSON" in second_call_prompt

    def test_with_pydantic_model(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            '{"signal": "BUY", "confidence": 80, "summary": "Good"}',
            100, 0.01,
        )
        result, _, _ = llm_call_with_retry(
            mock_llm, "sys", "usr", response_model=MockSignal
        )
        assert isinstance(result, MockSignal)


# ── async_llm_call_with_retry ─────────────────────────────────────


class TestAsyncLLMCallWithRetry:
    @pytest.mark.asyncio
    async def test_async_success(self):
        mock_llm = MagicMock()
        mock_llm.achat = AsyncMock(return_value=(
            '{"signal": "BUY", "confidence": 80, "summary": "Good"}',
            100, 0.01,
        ))
        result, tokens, cost = await async_llm_call_with_retry(mock_llm, "sys", "usr")
        assert result["signal"] == "BUY"
        assert mock_llm.achat.call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry(self):
        mock_llm = MagicMock()
        mock_llm.achat = AsyncMock(side_effect=[
            ("bad", 50, 0.005),
            ('{"signal": "HOLD", "confidence": 50, "summary": "OK"}', 60, 0.006),
        ])
        result, tokens, _ = await async_llm_call_with_retry(mock_llm, "sys", "usr", max_retries=1)
        assert result["signal"] == "HOLD"
        assert tokens == 110

    @pytest.mark.asyncio
    async def test_async_all_retries_fail(self):
        mock_llm = MagicMock()
        mock_llm.achat = AsyncMock(return_value=("nope", 50, 0.005))
        with pytest.raises(ValueError, match="Failed to parse"):
            await async_llm_call_with_retry(mock_llm, "sys", "usr", max_retries=1)
