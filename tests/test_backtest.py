"""Tests for portfolio.backtest module."""

from __future__ import annotations

import pytest

from trading_master.portfolio.backtest import (
    _is_correct,
    compute_agent_accuracy,
    compute_calibration,
    compute_hit_rate,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_outcome(
    action: str = "BUY",
    confidence: float = 70.0,
    return_30: float | None = None,
    return_90: float | None = None,
    return_180: float | None = None,
    analyst_reports: list[dict] | None = None,
    ticker: str = "TEST",
) -> dict:
    """Build a synthetic outcome dict matching track_recommendation_outcomes output."""
    outcomes: dict[int, dict] = {}
    for h, ret in [(30, return_30), (90, return_90), (180, return_180)]:
        if ret is not None:
            outcomes[h] = {
                "price": 100 + ret,
                "return_pct": ret,
                "correct": _is_correct(action, ret),
            }

    return {
        "id": 1,
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "rec_date": "2025-01-01T00:00:00",
        "rec_price": 100.0,
        "analyst_reports": analyst_reports or [],
        "outcomes": outcomes,
    }


# ── _is_correct ───────────────────────────────────────────────────────

class TestIsCorrect:
    def test_buy_positive_return(self):
        assert _is_correct("BUY", 5.0) is True

    def test_buy_negative_return(self):
        assert _is_correct("BUY", -3.0) is False

    def test_sell_negative_return(self):
        assert _is_correct("SELL", -5.0) is True

    def test_sell_positive_return(self):
        assert _is_correct("SELL", 5.0) is False

    def test_strong_buy_positive(self):
        assert _is_correct("STRONG_BUY", 2.0) is True

    def test_strong_sell_negative(self):
        assert _is_correct("STRONG_SELL", -2.0) is True

    def test_hold_small_move(self):
        assert _is_correct("HOLD", 2.0) is True

    def test_hold_large_move(self):
        assert _is_correct("HOLD", 10.0) is False

    def test_hold_boundary(self):
        assert _is_correct("HOLD", 5.0) is True
        assert _is_correct("HOLD", -5.0) is True

    def test_buy_zero_return(self):
        assert _is_correct("BUY", 0.0) is False

    def test_sell_zero_return(self):
        assert _is_correct("SELL", 0.0) is False


# ── compute_hit_rate ──────────────────────────────────────────────────

class TestComputeHitRate:
    def test_all_correct(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0),
            _make_outcome(action="BUY", return_90=5.0),
            _make_outcome(action="SELL", return_90=-8.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["hit_rate"] == 100.0
        assert hr["n_total"] == 3
        assert hr["n_correct"] == 3
        assert hr["n_buy"] == 2
        assert hr["n_sell"] == 1

    def test_none_correct(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=-10.0),
            _make_outcome(action="SELL", return_90=5.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["hit_rate"] == 0.0
        assert hr["n_correct"] == 0
        assert hr["n_total"] == 2

    def test_mixed(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0),
            _make_outcome(action="BUY", return_90=-5.0),
            _make_outcome(action="BUY", return_90=3.0),
            _make_outcome(action="BUY", return_90=-1.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["hit_rate"] == 50.0
        assert hr["n_correct"] == 2
        assert hr["n_total"] == 4

    def test_avg_return(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0),
            _make_outcome(action="BUY", return_90=-2.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["avg_return"] == 4.0  # (10 + -2) / 2

    def test_empty(self):
        hr = compute_hit_rate([], 90)
        assert hr["hit_rate"] is None
        assert hr["avg_return"] is None
        assert hr["n_total"] == 0

    def test_no_matching_horizon(self):
        outcomes = [_make_outcome(action="BUY", return_30=5.0)]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_total"] == 0
        assert hr["hit_rate"] is None

    def test_different_horizons(self):
        outcomes = [
            _make_outcome(action="BUY", return_30=10.0, return_90=5.0),
        ]
        hr30 = compute_hit_rate(outcomes, 30)
        hr90 = compute_hit_rate(outcomes, 90)
        assert hr30["avg_return"] == 10.0
        assert hr90["avg_return"] == 5.0

    def test_all_buy(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=5.0),
            _make_outcome(action="BUY", return_90=10.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_buy"] == 2
        assert hr["n_sell"] == 0

    def test_all_sell(self):
        outcomes = [
            _make_outcome(action="SELL", return_90=-5.0),
            _make_outcome(action="SELL", return_90=-10.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_buy"] == 0
        assert hr["n_sell"] == 2


# ── compute_agent_accuracy ────────────────────────────────────────────

class TestComputeAgentAccuracy:
    def test_single_agent(self):
        reports = [{"analyst": "fundamental", "signal": "BUY", "confidence": 80}]
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0, analyst_reports=reports),
        ]
        acc = compute_agent_accuracy(outcomes, 90)
        assert "fundamental" in acc
        assert acc["fundamental"]["hit_rate"] == 100.0
        assert acc["fundamental"]["avg_confidence"] == 80.0
        assert acc["fundamental"]["n"] == 1

    def test_multiple_agents(self):
        reports = [
            {"analyst": "fundamental", "signal": "BUY", "confidence": 80},
            {"analyst": "technical", "signal": "SELL", "confidence": 60},
            {"analyst": "sentiment", "signal": "BUY", "confidence": 70},
        ]
        outcomes = [
            _make_outcome(action="BUY", return_90=5.0, analyst_reports=reports),
        ]
        acc = compute_agent_accuracy(outcomes, 90)
        assert acc["fundamental"]["hit_rate"] == 100.0
        assert acc["technical"]["hit_rate"] == 0.0  # SELL signal but price went up
        assert acc["sentiment"]["hit_rate"] == 100.0

    def test_empty_reports(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0, analyst_reports=[]),
        ]
        acc = compute_agent_accuracy(outcomes, 90)
        assert acc == {}

    def test_no_outcomes_at_horizon(self):
        reports = [{"analyst": "fundamental", "signal": "BUY", "confidence": 80}]
        outcomes = [
            _make_outcome(action="BUY", return_30=10.0, analyst_reports=reports),
        ]
        acc = compute_agent_accuracy(outcomes, 90)
        assert acc == {}

    def test_agent_mixed_results(self):
        reports = [{"analyst": "fundamental", "signal": "BUY", "confidence": 75}]
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0, analyst_reports=reports),
            _make_outcome(action="BUY", return_90=-5.0, analyst_reports=reports),
        ]
        acc = compute_agent_accuracy(outcomes, 90)
        assert acc["fundamental"]["hit_rate"] == 50.0
        assert acc["fundamental"]["n"] == 2
        assert acc["fundamental"]["avg_confidence"] == 75.0

    def test_empty_outcomes(self):
        acc = compute_agent_accuracy([], 90)
        assert acc == {}


# ── compute_calibration ──────────────────────────────────────────────

class TestComputeCalibration:
    def test_single_bucket(self):
        outcomes = [
            _make_outcome(confidence=80, action="BUY", return_90=5.0),
            _make_outcome(confidence=85, action="BUY", return_90=10.0),
            _make_outcome(confidence=90, action="BUY", return_90=-3.0),
        ]
        cal = compute_calibration(outcomes, 90)
        # All in the 75-100 bucket
        high_bucket = [b for b in cal if b["bucket"] == "75-100"][0]
        assert high_bucket["count"] == 3
        assert high_bucket["hit_rate"] == pytest.approx(66.67, abs=0.01)
        assert high_bucket["avg_confidence"] == pytest.approx(85.0, abs=0.01)

    def test_multiple_buckets(self):
        outcomes = [
            _make_outcome(confidence=20, action="BUY", return_90=5.0),
            _make_outcome(confidence=40, action="BUY", return_90=5.0),
            _make_outcome(confidence=60, action="BUY", return_90=5.0),
            _make_outcome(confidence=80, action="BUY", return_90=5.0),
        ]
        cal = compute_calibration(outcomes, 90)
        buckets = {b["bucket"]: b for b in cal}
        assert buckets["0-25"]["count"] == 1
        assert buckets["25-50"]["count"] == 1
        assert buckets["50-75"]["count"] == 1
        assert buckets["75-100"]["count"] == 1

    def test_empty_outcomes(self):
        cal = compute_calibration([], 90)
        assert len(cal) == 4
        for b in cal:
            assert b["count"] == 0
            assert b["hit_rate"] is None

    def test_all_in_one_bucket(self):
        outcomes = [
            _make_outcome(confidence=55, action="BUY", return_90=5.0),
            _make_outcome(confidence=60, action="BUY", return_90=-2.0),
            _make_outcome(confidence=65, action="BUY", return_90=8.0),
            _make_outcome(confidence=70, action="BUY", return_90=1.0),
        ]
        cal = compute_calibration(outcomes, 90)
        buckets = {b["bucket"]: b for b in cal}
        assert buckets["50-75"]["count"] == 4
        assert buckets["50-75"]["hit_rate"] == 75.0  # 3 correct out of 4
        assert buckets["0-25"]["count"] == 0
        assert buckets["25-50"]["count"] == 0
        assert buckets["75-100"]["count"] == 0

    def test_no_matching_horizon(self):
        outcomes = [
            _make_outcome(confidence=80, action="BUY", return_30=5.0),
        ]
        cal = compute_calibration(outcomes, 90)
        for b in cal:
            assert b["count"] == 0

    def test_perfect_calibration(self):
        # 100% hit rate in a bucket should show 100%
        outcomes = [
            _make_outcome(confidence=80, action="BUY", return_90=5.0),
            _make_outcome(confidence=85, action="BUY", return_90=10.0),
        ]
        cal = compute_calibration(outcomes, 90)
        high_bucket = [b for b in cal if b["bucket"] == "75-100"][0]
        assert high_bucket["hit_rate"] == 100.0


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_recommendations(self):
        hr = compute_hit_rate([], 90)
        assert hr["n_total"] == 0
        assert hr["hit_rate"] is None

        acc = compute_agent_accuracy([], 90)
        assert acc == {}

        cal = compute_calibration([], 90)
        assert all(b["count"] == 0 for b in cal)

    def test_all_buy_outcomes(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=10.0, confidence=80),
            _make_outcome(action="BUY", return_90=5.0, confidence=60),
            _make_outcome(action="BUY", return_90=-2.0, confidence=40),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_buy"] == 3
        assert hr["n_sell"] == 0
        assert hr["hit_rate"] == pytest.approx(66.67, abs=0.01)

    def test_all_sell_outcomes(self):
        outcomes = [
            _make_outcome(action="SELL", return_90=-10.0, confidence=80),
            _make_outcome(action="SELL", return_90=-5.0, confidence=60),
            _make_outcome(action="SELL", return_90=2.0, confidence=40),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_buy"] == 0
        assert hr["n_sell"] == 3
        assert hr["hit_rate"] == pytest.approx(66.67, abs=0.01)

    def test_hold_actions(self):
        outcomes = [
            _make_outcome(action="HOLD", return_90=2.0),
            _make_outcome(action="HOLD", return_90=-3.0),
            _make_outcome(action="HOLD", return_90=15.0),  # too big, incorrect for HOLD
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["n_total"] == 3
        assert hr["n_correct"] == 2  # first two within +/-5%

    def test_single_recommendation(self):
        outcomes = [_make_outcome(action="BUY", return_90=5.0, confidence=75)]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["hit_rate"] == 100.0
        assert hr["n_total"] == 1

    def test_extreme_returns(self):
        outcomes = [
            _make_outcome(action="BUY", return_90=200.0),
            _make_outcome(action="SELL", return_90=-80.0),
        ]
        hr = compute_hit_rate(outcomes, 90)
        assert hr["hit_rate"] == 100.0
        assert hr["avg_return"] == 60.0  # (200 + -80) / 2
