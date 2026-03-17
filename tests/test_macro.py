"""Tests for the macro data module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_master.models import MacroData, MarketRegime
from trading_master.data.macro import (
    _classify_vix,
    _detect_regime,
    _build_summary,
    fetch_macro_data,
)


# ── MacroData model defaults ─────────────────────────────────────────

class TestMacroDataModel:
    def test_defaults(self):
        m = MacroData()
        assert m.us_10yr_yield is None
        assert m.us_2yr_yield is None
        assert m.yield_curve_spread is None
        assert m.yield_curve_inverted is False
        assert m.vix is None
        assert m.vix_regime == "normal"
        assert m.sp500_price is None
        assert m.sp500_sma200 is None
        assert m.sp500_above_sma200 is True
        assert m.regime == MarketRegime.SIDEWAYS
        assert m.regime_signals == []
        assert m.summary == ""

    def test_with_values(self):
        m = MacroData(
            us_10yr_yield=4.5,
            vix=22.0,
            sp500_price=5100.0,
            sp500_sma200=4900.0,
            regime=MarketRegime.BULL,
        )
        assert m.us_10yr_yield == 4.5
        assert m.vix == 22.0
        assert m.regime == MarketRegime.BULL

    def test_serialization_roundtrip(self):
        m = MacroData(
            us_10yr_yield=4.2,
            vix=18.5,
            regime=MarketRegime.BEAR,
            regime_signals=["test signal"],
        )
        d = m.model_dump(mode="json")
        m2 = MacroData(**d)
        assert m2.us_10yr_yield == m.us_10yr_yield
        assert m2.vix == m.vix
        assert m2.regime == m.regime
        assert m2.regime_signals == ["test signal"]


# ── VIX regime classification ────────────────────────────────────────

class TestVixClassification:
    def test_none(self):
        assert _classify_vix(None) == "normal"

    def test_low(self):
        assert _classify_vix(10.0) == "low"
        assert _classify_vix(14.9) == "low"

    def test_normal(self):
        assert _classify_vix(15.0) == "normal"
        assert _classify_vix(20.0) == "normal"
        assert _classify_vix(25.0) == "normal"

    def test_high(self):
        assert _classify_vix(25.1) == "high"
        assert _classify_vix(30.0) == "high"
        assert _classify_vix(35.0) == "high"

    def test_extreme(self):
        assert _classify_vix(35.1) == "extreme"
        assert _classify_vix(80.0) == "extreme"


# ── Regime detection ─────────────────────────────────────────────────

class TestRegimeDetection:
    def test_crisis_high_vix_below_sma(self):
        regime, signals = _detect_regime(vix=40.0, sp500_above_sma200=False, yield_curve_inverted=False)
        assert regime == MarketRegime.CRISIS

    def test_bear_below_sma_inverted_curve(self):
        regime, signals = _detect_regime(vix=22.0, sp500_above_sma200=False, yield_curve_inverted=True)
        assert regime == MarketRegime.BEAR

    def test_bull_above_sma_low_vix(self):
        regime, signals = _detect_regime(vix=15.0, sp500_above_sma200=True, yield_curve_inverted=False)
        assert regime == MarketRegime.BULL

    def test_sideways_default(self):
        regime, signals = _detect_regime(vix=22.0, sp500_above_sma200=True, yield_curve_inverted=False)
        assert regime == MarketRegime.SIDEWAYS

    def test_sideways_mixed_signals(self):
        # Above SMA but VIX is high — not quite bull, not bear
        regime, signals = _detect_regime(vix=28.0, sp500_above_sma200=True, yield_curve_inverted=True)
        assert regime == MarketRegime.SIDEWAYS

    def test_crisis_takes_priority_over_bear(self):
        # VIX > 35 AND below SMA AND inverted — should be CRISIS not BEAR
        regime, signals = _detect_regime(vix=45.0, sp500_above_sma200=False, yield_curve_inverted=True)
        assert regime == MarketRegime.CRISIS

    def test_none_vix(self):
        # Without VIX data, can't be CRISIS or BULL (VIX-dependent)
        regime, signals = _detect_regime(vix=None, sp500_above_sma200=False, yield_curve_inverted=True)
        assert regime == MarketRegime.BEAR

    def test_none_vix_above_sma(self):
        regime, signals = _detect_regime(vix=None, sp500_above_sma200=True, yield_curve_inverted=False)
        assert regime == MarketRegime.SIDEWAYS

    def test_signals_populated(self):
        _, signals = _detect_regime(vix=40.0, sp500_above_sma200=False, yield_curve_inverted=True)
        assert any("VIX" in s for s in signals)
        assert any("below" in s.lower() for s in signals)
        assert any("inverted" in s.lower() for s in signals)


# ── Summary builder ──────────────────────────────────────────────────

class TestBuildSummary:
    def test_summary_contains_regime(self):
        m = MacroData(regime=MarketRegime.BULL)
        summary = _build_summary(m)
        assert "BULL" in summary

    def test_summary_contains_vix(self):
        m = MacroData(vix=22.5, vix_regime="normal")
        summary = _build_summary(m)
        assert "22.5" in summary
        assert "normal" in summary

    def test_summary_inverted_curve(self):
        m = MacroData(yield_curve_spread=-0.5, yield_curve_inverted=True)
        summary = _build_summary(m)
        assert "INVERTED" in summary


# ── fetch_macro_data with mocked yfinance ────────────────────────────

class TestFetchMacroData:
    @patch("trading_master.data.macro.get_db")
    @patch("trading_master.data.macro.yf")
    def test_fetch_with_cache_hit(self, mock_yf, mock_get_db):
        """When cache has data, return it without calling yfinance."""
        mock_db = MagicMock()
        mock_db.cache_get.return_value = MacroData(
            vix=20.0, regime=MarketRegime.BULL
        ).model_dump(mode="json")
        mock_get_db.return_value = mock_db

        result = fetch_macro_data()
        assert result.vix == 20.0
        assert result.regime == MarketRegime.BULL
        mock_yf.Ticker.assert_not_called()

    @patch("trading_master.data.macro.get_db")
    @patch("trading_master.data.macro._safe_price")
    @patch("trading_master.data.macro.yf")
    def test_fetch_cache_miss_bull(self, mock_yf, mock_safe_price, mock_get_db):
        """Cache miss — fetch from yfinance and detect bull regime."""
        import pandas as pd
        import numpy as np

        mock_db = MagicMock()
        mock_db.cache_get.return_value = None
        mock_get_db.return_value = mock_db

        # _safe_price returns: ^TNX=4.5, ^IRX=4.0, ^VIX=15.0
        mock_safe_price.side_effect = lambda t: {
            "^TNX": 4.5, "^IRX": 4.0, "^VIX": 15.0,
        }.get(t)

        # SPY history: 250 days, price trending up, last close = 5200
        dates = pd.date_range(end="2025-01-15", periods=250, freq="B")
        prices = np.linspace(4800, 5200, 250)
        spy_hist = pd.DataFrame({"Close": prices}, index=dates)

        mock_spy_ticker = MagicMock()
        mock_spy_ticker.history.return_value = spy_hist
        mock_yf.Ticker.return_value = mock_spy_ticker

        result = fetch_macro_data()

        assert result.us_10yr_yield == 4.5
        assert result.us_2yr_yield == 4.0
        assert result.vix == 15.0
        assert result.yield_curve_spread == 0.5
        assert result.yield_curve_inverted is False
        assert result.sp500_above_sma200 is True
        assert result.regime == MarketRegime.BULL
        assert result.vix_regime == "normal"

        # Verify caching
        mock_db.cache_set.assert_called_once()

    @patch("trading_master.data.macro.get_db")
    @patch("trading_master.data.macro._safe_price")
    @patch("trading_master.data.macro.yf")
    def test_fetch_cache_miss_crisis(self, mock_yf, mock_safe_price, mock_get_db):
        """Detect crisis when VIX extreme and S&P below SMA200."""
        import pandas as pd
        import numpy as np

        mock_db = MagicMock()
        mock_db.cache_get.return_value = None
        mock_get_db.return_value = mock_db

        mock_safe_price.side_effect = lambda t: {
            "^TNX": 3.0, "^IRX": 4.0, "^VIX": 45.0,
        }.get(t)

        # SPY crashing — last price well below SMA200
        dates = pd.date_range(end="2025-01-15", periods=250, freq="B")
        prices = np.linspace(5200, 4200, 250)  # trending down
        spy_hist = pd.DataFrame({"Close": prices}, index=dates)

        mock_spy_ticker = MagicMock()
        mock_spy_ticker.history.return_value = spy_hist
        mock_yf.Ticker.return_value = mock_spy_ticker

        result = fetch_macro_data()

        assert result.vix == 45.0
        assert result.vix_regime == "extreme"
        assert result.yield_curve_inverted is True  # 3.0 - 4.0 = -1.0
        assert result.sp500_above_sma200 is False
        assert result.regime == MarketRegime.CRISIS
