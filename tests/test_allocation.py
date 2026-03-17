"""Tests for the asset allocation framework."""

from __future__ import annotations

import pytest

from trading_master.models import (
    AssetClass,
    AllocationModel,
    AllocationTarget,
    Position,
    PortfolioState,
)
from trading_master.portfolio.allocation import (
    PRESET_MODELS,
    TICKER_CLASS_MAP,
    REGIME_ADJUSTMENTS,
    OVERLAP_GROUPS,
    classify_ticker,
    compute_current_allocation,
    compute_drift,
    needs_rebalance,
    suggest_rebalance,
    regime_adjusted_model,
    regime_allocation_alert,
    detect_overlaps,
    effective_concentration,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_portfolio(positions_dict: dict[str, tuple[float, float, float]], cash: float = 10000.0) -> PortfolioState:
    """Build a PortfolioState from {ticker: (quantity, avg_cost, current_price)}."""
    positions = {}
    for ticker, (qty, avg_cost, price) in positions_dict.items():
        pos = Position(ticker=ticker, quantity=qty, avg_cost=avg_cost)
        pos.update_market(price)
        positions[ticker] = pos
    state = PortfolioState(positions=positions, cash=cash)
    state.recalculate()
    return state


# ── classify_ticker ───────────────────────────────────────────────────

class TestClassifyTicker:
    def test_known_us_equity(self):
        assert classify_ticker("SPY") == AssetClass.US_EQUITY
        assert classify_ticker("MSFT") == AssetClass.US_EQUITY
        assert classify_ticker("QQQ") == AssetClass.US_EQUITY

    def test_known_fixed_income(self):
        assert classify_ticker("TLT") == AssetClass.FIXED_INCOME
        assert classify_ticker("BND") == AssetClass.FIXED_INCOME

    def test_known_short_term_treasury(self):
        assert classify_ticker("SGOV") == AssetClass.SHORT_TERM_TREASURY

    def test_known_commodities(self):
        assert classify_ticker("GLD") == AssetClass.COMMODITIES
        assert classify_ticker("COPX") == AssetClass.COMMODITIES

    def test_known_alternatives(self):
        assert classify_ticker("ARKVX") == AssetClass.ALTERNATIVES
        assert classify_ticker("RVI") == AssetClass.ALTERNATIVES

    def test_unknown_defaults_to_us_equity(self):
        assert classify_ticker("ZZZZZ") == AssetClass.US_EQUITY
        assert classify_ticker("FAKE") == AssetClass.US_EQUITY

    def test_case_insensitive(self):
        assert classify_ticker("spy") == AssetClass.US_EQUITY
        assert classify_ticker("Tlt") == AssetClass.FIXED_INCOME


# ── compute_current_allocation ────────────────────────────────────────

class TestComputeCurrentAllocation:
    def test_simple_portfolio(self):
        # $50k equity + $20k bonds + $10k cash = $80k total
        state = _make_portfolio(
            {"SPY": (100, 400, 500), "TLT": (200, 90, 100)},
            cash=10000,
        )
        alloc = compute_current_allocation(state)
        alloc_map = {a.asset_class: a.current_pct for a in alloc}

        total = 100 * 500 + 200 * 100 + 10000  # 80000
        assert total == state.total_value

        expected_equity = (50000 / 80000) * 100
        expected_fi = (20000 / 80000) * 100
        expected_cash = (10000 / 80000) * 100

        assert abs(alloc_map[AssetClass.US_EQUITY] - expected_equity) < 0.1
        assert abs(alloc_map[AssetClass.FIXED_INCOME] - expected_fi) < 0.1
        assert abs(alloc_map[AssetClass.CASH] - expected_cash) < 0.1

    def test_empty_portfolio(self):
        state = PortfolioState(positions={}, cash=0, total_value=0)
        alloc = compute_current_allocation(state)
        assert alloc == []

    def test_cash_only(self):
        state = _make_portfolio({}, cash=100000)
        alloc = compute_current_allocation(state)
        alloc_map = {a.asset_class: a.current_pct for a in alloc}
        assert abs(alloc_map[AssetClass.CASH] - 100.0) < 0.1

    def test_all_asset_classes_present(self):
        state = _make_portfolio({"SPY": (10, 100, 100)}, cash=1000)
        alloc = compute_current_allocation(state)
        classes_present = {a.asset_class for a in alloc}
        # Should have all AssetClass values
        for ac in AssetClass:
            assert ac in classes_present


# ── compute_drift ─────────────────────────────────────────────────────

class TestComputeDrift:
    def test_balanced_model_drift(self):
        # Portfolio: 80% equity, 5% fixed income, 15% cash
        # vs balanced: 50% equity, 20% fixed income, 5% cash
        state = _make_portfolio(
            {"SPY": (160, 500, 500), "TLT": (50, 100, 100)},
            cash=15000,
        )
        # total = 80000 + 5000 + 15000 = 100000
        current = compute_current_allocation(state)
        model = PRESET_MODELS["balanced"]
        drift = compute_drift(current, model)
        drift_map = {d.asset_class: d for d in drift}

        # US equity: 80% current vs 50% target = +30% drift
        assert drift_map[AssetClass.US_EQUITY].drift_pct == pytest.approx(30.0, abs=0.1)
        # Fixed income: 5% current vs 20% target = -15% drift
        assert drift_map[AssetClass.FIXED_INCOME].drift_pct == pytest.approx(-15.0, abs=0.1)
        # Cash: 15% current vs 5% target = +10% drift
        assert drift_map[AssetClass.CASH].drift_pct == pytest.approx(10.0, abs=0.1)

    def test_perfect_allocation_zero_drift(self):
        # Build a portfolio that exactly matches balanced model
        # balanced: 50% equity, 20% FI, 15% ST treasury, 5% commodities, 5% alternatives, 5% cash
        total = 100000
        state = _make_portfolio(
            {
                "SPY": (100, 500, 500),     # $50,000 = 50%
                "TLT": (200, 100, 100),     # $20,000 = 20%
                "SGOV": (150, 100, 100),    # $15,000 = 15%
                "GLD": (25, 200, 200),      # $5,000 = 5%
                "RVI": (50, 100, 100),      # $5,000 = 5%
            },
            cash=5000,                       # $5,000 = 5%
        )
        current = compute_current_allocation(state)
        model = PRESET_MODELS["balanced"]
        drift = compute_drift(current, model)

        for d in drift:
            assert abs(d.drift_pct) < 0.1, f"{d.asset_class} drift {d.drift_pct} is not ~0"


# ── needs_rebalance ──────────────────────────────────────────────────

class TestNeedsRebalance:
    def test_no_rebalance_within_threshold(self):
        targets = [
            AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=50, min_pct=40, max_pct=60, current_pct=52, drift_pct=2.0),
            AllocationTarget(asset_class=AssetClass.CASH, target_pct=5, min_pct=2, max_pct=15, current_pct=3, drift_pct=-2.0),
        ]
        assert needs_rebalance(targets, threshold_pct=5.0) is False

    def test_rebalance_needed_over_threshold(self):
        targets = [
            AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=50, min_pct=40, max_pct=60, current_pct=60, drift_pct=10.0),
            AllocationTarget(asset_class=AssetClass.CASH, target_pct=5, min_pct=2, max_pct=15, current_pct=3, drift_pct=-2.0),
        ]
        assert needs_rebalance(targets, threshold_pct=5.0) is True

    def test_negative_drift_triggers(self):
        targets = [
            AllocationTarget(asset_class=AssetClass.FIXED_INCOME, target_pct=20, min_pct=15, max_pct=30, current_pct=8, drift_pct=-12.0),
        ]
        assert needs_rebalance(targets, threshold_pct=5.0) is True

    def test_exact_threshold_does_not_trigger(self):
        targets = [
            AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=50, min_pct=40, max_pct=60, current_pct=55, drift_pct=5.0),
        ]
        assert needs_rebalance(targets, threshold_pct=5.0) is False

    def test_custom_threshold(self):
        targets = [
            AllocationTarget(asset_class=AssetClass.US_EQUITY, target_pct=50, min_pct=40, max_pct=60, current_pct=54, drift_pct=4.0),
        ]
        assert needs_rebalance(targets, threshold_pct=3.0) is True
        assert needs_rebalance(targets, threshold_pct=5.0) is False


# ── suggest_rebalance ─────────────────────────────────────────────────

class TestSuggestRebalance:
    def test_overweight_equity_suggests_sell(self):
        # 80% equity, balanced wants 50% => should suggest SELL equity
        state = _make_portfolio(
            {"SPY": (160, 500, 500), "TLT": (50, 100, 100)},
            cash=15000,
        )
        model = PRESET_MODELS["balanced"]
        suggestions = suggest_rebalance(state, model)

        equity_sug = [s for s in suggestions if s["asset_class"] == "us_equity"]
        assert len(equity_sug) == 1
        assert equity_sug[0]["direction"] == "SELL"
        assert equity_sug[0]["amount_usd"] > 0

    def test_underweight_fi_suggests_buy(self):
        # 80% equity, 5% FI, balanced wants 20% FI => should suggest BUY FI
        state = _make_portfolio(
            {"SPY": (160, 500, 500), "TLT": (50, 100, 100)},
            cash=15000,
        )
        model = PRESET_MODELS["balanced"]
        suggestions = suggest_rebalance(state, model)

        fi_sug = [s for s in suggestions if s["asset_class"] == "fixed_income"]
        assert len(fi_sug) == 1
        assert fi_sug[0]["direction"] == "BUY"
        assert fi_sug[0]["amount_usd"] > 0

    def test_balanced_portfolio_no_suggestions(self):
        state = _make_portfolio(
            {
                "SPY": (100, 500, 500),
                "TLT": (200, 100, 100),
                "SGOV": (150, 100, 100),
                "GLD": (25, 200, 200),
                "RVI": (50, 100, 100),
            },
            cash=5000,
        )
        model = PRESET_MODELS["balanced"]
        suggestions = suggest_rebalance(state, model)
        # Should have no suggestions (or only trivial ones)
        assert all(abs(s["drift_pct"]) < 1.0 for s in suggestions) or len(suggestions) == 0

    def test_suggestions_sorted_by_drift(self):
        state = _make_portfolio(
            {"SPY": (160, 500, 500), "TLT": (50, 100, 100)},
            cash=15000,
        )
        model = PRESET_MODELS["balanced"]
        suggestions = suggest_rebalance(state, model)

        # Should be sorted by absolute drift descending
        drifts = [abs(s["drift_pct"]) for s in suggestions]
        assert drifts == sorted(drifts, reverse=True)

    def test_suggestions_have_tickers(self):
        state = _make_portfolio(
            {"SPY": (160, 500, 500)},
            cash=15000,
        )
        model = PRESET_MODELS["balanced"]
        suggestions = suggest_rebalance(state, model)

        fi_sug = [s for s in suggestions if s["asset_class"] == "fixed_income"]
        if fi_sug:
            assert len(fi_sug[0]["suggested_tickers"]) > 0


# ── PRESET_MODELS validation ─────────────────────────────────────────

class TestPresetModels:
    @pytest.mark.parametrize("name", ["balanced", "growth", "conservative"])
    def test_targets_sum_to_100(self, name):
        model = PRESET_MODELS[name]
        total = sum(t.target_pct for t in model.targets)
        assert total == pytest.approx(100.0, abs=0.1), f"{name} targets sum to {total}"

    @pytest.mark.parametrize("name", ["balanced", "growth", "conservative"])
    def test_min_less_than_max(self, name):
        model = PRESET_MODELS[name]
        for t in model.targets:
            assert t.min_pct <= t.max_pct, f"{name}/{t.asset_class}: min > max"

    @pytest.mark.parametrize("name", ["balanced", "growth", "conservative"])
    def test_target_within_range(self, name):
        model = PRESET_MODELS[name]
        for t in model.targets:
            assert t.min_pct <= t.target_pct <= t.max_pct, (
                f"{name}/{t.asset_class}: target {t.target_pct} not in [{t.min_pct}, {t.max_pct}]"
            )


# ── regime_adjusted_model ────────────────────────────────────────────

class TestRegimeAdjustedModel:
    def test_bull_no_change(self):
        base = PRESET_MODELS["balanced"]
        adjusted = regime_adjusted_model(base, "bull")
        # Bull has no adjustments, should return same model
        assert adjusted is base

    def test_sideways_reduces_equity(self):
        base = PRESET_MODELS["balanced"]
        adjusted = regime_adjusted_model(base, "sideways")
        target_map = {t.asset_class: t for t in adjusted.targets}
        base_map = {t.asset_class: t for t in base.targets}
        # US equity should be reduced by 5
        assert target_map[AssetClass.US_EQUITY].target_pct == base_map[AssetClass.US_EQUITY].target_pct - 5
        # Short-term treasury should increase by 3
        assert target_map[AssetClass.SHORT_TERM_TREASURY].target_pct == base_map[AssetClass.SHORT_TERM_TREASURY].target_pct + 3
        # Cash should increase by 2
        assert target_map[AssetClass.CASH].target_pct == base_map[AssetClass.CASH].target_pct + 2

    def test_bear_large_equity_reduction(self):
        base = PRESET_MODELS["balanced"]
        adjusted = regime_adjusted_model(base, "bear")
        target_map = {t.asset_class: t for t in adjusted.targets}
        base_map = {t.asset_class: t for t in base.targets}
        assert target_map[AssetClass.US_EQUITY].target_pct == base_map[AssetClass.US_EQUITY].target_pct - 15
        assert target_map[AssetClass.FIXED_INCOME].target_pct == base_map[AssetClass.FIXED_INCOME].target_pct + 5
        assert target_map[AssetClass.SHORT_TERM_TREASURY].target_pct == base_map[AssetClass.SHORT_TERM_TREASURY].target_pct + 7
        assert target_map[AssetClass.CASH].target_pct == base_map[AssetClass.CASH].target_pct + 3

    def test_crisis_max_defensive(self):
        base = PRESET_MODELS["balanced"]
        adjusted = regime_adjusted_model(base, "crisis")
        target_map = {t.asset_class: t for t in adjusted.targets}
        base_map = {t.asset_class: t for t in base.targets}
        assert target_map[AssetClass.US_EQUITY].target_pct == base_map[AssetClass.US_EQUITY].target_pct - 25
        assert target_map[AssetClass.CASH].target_pct == base_map[AssetClass.CASH].target_pct + 8

    def test_adjusted_name_includes_regime(self):
        adjusted = regime_adjusted_model(PRESET_MODELS["growth"], "bear")
        assert "bear" in adjusted.name
        assert "growth" in adjusted.name

    def test_min_max_constraints_valid(self):
        """Adjusted model should always have min <= target <= max."""
        for regime in ["bull", "sideways", "bear", "crisis"]:
            for model_name in ["balanced", "growth", "conservative"]:
                adjusted = regime_adjusted_model(PRESET_MODELS[model_name], regime)
                for t in adjusted.targets:
                    assert t.min_pct <= t.target_pct <= t.max_pct, (
                        f"{model_name}/{regime}/{t.asset_class}: target {t.target_pct} "
                        f"not in [{t.min_pct}, {t.max_pct}]"
                    )

    def test_target_pct_clamped_non_negative(self):
        """Even with large reductions, target_pct should not go below 0."""
        # Conservative has only 30% equity; crisis reduces by 25 => 5% (still valid)
        adjusted = regime_adjusted_model(PRESET_MODELS["conservative"], "crisis")
        for t in adjusted.targets:
            assert t.target_pct >= 0, f"{t.asset_class} target_pct is negative: {t.target_pct}"


# ── regime_allocation_alert ──────────────────────────────────────────

class TestRegimeAllocationAlert:
    def test_bear_overweight_equity_alert(self):
        # 76% equity in a bear market (adjusted target ~35% for balanced)
        state = _make_portfolio(
            {"SPY": (152, 500, 500)},  # $76,000
            cash=24000,                 # $24,000
        )
        # total = $100,000; equity = 76%, cash = 24%
        alerts = regime_allocation_alert(state, PRESET_MODELS["balanced"], "bear")
        # Should flag equity as too high
        equity_alerts = [a for a in alerts if "Us Equity" in a]
        assert len(equity_alerts) >= 1
        assert "BEAR" in equity_alerts[0]
        assert "exceeds" in equity_alerts[0]

    def test_bull_no_alerts_for_normal_portfolio(self):
        # Balanced portfolio in bull market should have no alerts
        state = _make_portfolio(
            {
                "SPY": (100, 500, 500),
                "TLT": (200, 100, 100),
                "SGOV": (150, 100, 100),
                "GLD": (25, 200, 200),
                "RVI": (50, 100, 100),
            },
            cash=5000,
        )
        alerts = regime_allocation_alert(state, PRESET_MODELS["balanced"], "bull")
        assert len(alerts) == 0

    def test_crisis_generates_multiple_alerts(self):
        # Heavy equity portfolio in crisis should generate alerts
        state = _make_portfolio(
            {"SPY": (180, 500, 500)},  # $90,000 = 90% equity
            cash=10000,
        )
        alerts = regime_allocation_alert(state, PRESET_MODELS["balanced"], "crisis")
        assert len(alerts) >= 1
        # Should mention CRISIS
        assert any("CRISIS" in a for a in alerts)

    def test_alert_format(self):
        state = _make_portfolio(
            {"SPY": (160, 500, 500)},  # $80,000 = ~80%
            cash=20000,
        )
        alerts = regime_allocation_alert(state, PRESET_MODELS["balanced"], "bear")
        for alert in alerts:
            # Each alert should mention the regime and contain "range"
            assert "BEAR" in alert
            assert "range" in alert


# ── detect_overlaps ─────────────────────────────────────────────────

class TestDetectOverlaps:
    def test_spy_voo_overlap(self):
        positions = {"SPY": 50000.0, "VOO": 30000.0, "TLT": 20000.0}
        overlaps = detect_overlaps(positions)
        sp500 = [o for o in overlaps if o["group"] == "sp500_trackers"]
        assert len(sp500) == 1
        assert set(sp500[0]["tickers"]) == {"SPY", "VOO"}
        assert sp500[0]["combined_pct"] == pytest.approx(80.0, abs=0.1)
        assert "VOO" in sp500[0]["suggestion"]

    def test_qqq_qqqm_overlap(self):
        positions = {"QQQ": 40000.0, "QQQM": 10000.0, "BND": 50000.0}
        overlaps = detect_overlaps(positions)
        nasdaq = [o for o in overlaps if o["group"] == "nasdaq100_trackers"]
        assert len(nasdaq) == 1
        assert set(nasdaq[0]["tickers"]) == {"QQQ", "QQQM"}
        assert nasdaq[0]["combined_pct"] == pytest.approx(50.0, abs=0.1)

    def test_no_overlap_single_ticker_per_group(self):
        positions = {"SPY": 50000.0, "QQQ": 30000.0, "TLT": 20000.0}
        overlaps = detect_overlaps(positions)
        assert len(overlaps) == 0

    def test_multiple_overlaps(self):
        positions = {
            "SPY": 30000.0, "VOO": 20000.0,  # sp500 overlap
            "QQQ": 15000.0, "QQQM": 10000.0,  # nasdaq overlap
            "TLT": 25000.0,
        }
        overlaps = detect_overlaps(positions)
        groups = {o["group"] for o in overlaps}
        assert "sp500_trackers" in groups
        assert "nasdaq100_trackers" in groups

    def test_empty_positions(self):
        overlaps = detect_overlaps({})
        assert overlaps == []

    def test_short_treasury_overlap(self):
        positions = {"SGOV": 30000.0, "SHV": 20000.0, "SPY": 50000.0}
        overlaps = detect_overlaps(positions)
        st = [o for o in overlaps if o["group"] == "short_treasury"]
        assert len(st) == 1
        assert set(st[0]["tickers"]) == {"SGOV", "SHV"}


# ── effective_concentration ──────────────────────────────────────────

class TestEffectiveConcentration:
    def test_overlaps_merged_as_single_position(self):
        # SPY + VOO should be treated as one position
        positions = {"SPY": 50000.0, "VOO": 30000.0, "TLT": 20000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        # Should have 2 effective positions (sp500_trackers + TLT)
        assert result["effective_positions"] == 2

    def test_no_overlap_all_distinct(self):
        positions = {"SPY": 40000.0, "TLT": 30000.0, "GLD": 30000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        assert result["effective_positions"] == 3

    def test_herfindahl_calculation(self):
        # Equal weights: 50/50 => HHI = 50^2 + 50^2 = 5000
        positions = {"SPY": 50000.0, "TLT": 50000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        assert result["herfindahl"] == pytest.approx(5000.0, abs=1.0)

    def test_top3_pct(self):
        positions = {"SPY": 50000.0, "VOO": 20000.0, "TLT": 15000.0, "GLD": 15000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        # Effective: sp500_trackers=70k, TLT=15k, GLD=15k => top3 = 100%
        assert result["top3_pct"] == pytest.approx(100.0, abs=0.1)

    def test_overlap_warnings_included(self):
        positions = {"SPY": 50000.0, "VOO": 30000.0, "TLT": 20000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        assert len(result["overlap_warnings"]) >= 1
        assert result["overlap_warnings"][0]["group"] == "sp500_trackers"

    def test_empty_portfolio(self):
        result = effective_concentration({}, 0.0)
        assert result["effective_positions"] == 0
        assert result["herfindahl"] == 0.0
        assert result["top3_pct"] == 0.0

    def test_single_position(self):
        positions = {"SPY": 100000.0}
        total = 100000.0
        result = effective_concentration(positions, total)
        assert result["effective_positions"] == 1
        assert result["herfindahl"] == pytest.approx(10000.0, abs=1.0)
        assert result["top3_pct"] == pytest.approx(100.0, abs=0.1)
