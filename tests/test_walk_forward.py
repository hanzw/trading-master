"""Tests for the walk-forward backtest module using synthetic data (no yfinance)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from trading_master.portfolio.walk_forward import (
    walk_forward_test,
    _compute_hurst,
    _compute_atr,
    _compute_avg_correlation,
    _annualized_sharpe,
)
from trading_master.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers: generate synthetic price data
# ---------------------------------------------------------------------------

def _make_gbm_prices(
    n_days: int,
    mu: float = 0.0005,
    sigma: float = 0.02,
    start_price: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic prices using geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, n_days - 1)
    log_prices = np.cumsum(np.concatenate([[np.log(start_price)], returns]))
    return np.exp(log_prices)


def _synthetic_fetcher(tickers: list[str], days: int) -> dict[str, np.ndarray]:
    """Mock data fetcher returning synthetic GBM prices per ticker."""
    result = {}
    for i, t in enumerate(tickers):
        result[t] = _make_gbm_prices(days, seed=42 + i)
    return result


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestComputeHurst:
    def test_random_walk_hurst_near_05(self):
        """Random walk should yield Hurst near 0.5."""
        prices = _make_gbm_prices(500, mu=0.0, sigma=0.02, seed=123)
        h = _compute_hurst(prices)
        assert 0.3 <= h <= 0.7, f"Hurst {h} outside expected range for random walk"

    def test_short_series_returns_default(self):
        """Very short series should return default 0.5."""
        assert _compute_hurst(np.array([100.0, 101.0, 102.0])) == 0.5

    def test_returns_float(self):
        prices = _make_gbm_prices(300, seed=99)
        h = _compute_hurst(prices)
        assert isinstance(h, float)
        assert 0.01 <= h <= 0.99


class TestComputeATR:
    def test_basic_atr(self):
        prices = np.array([100, 102, 101, 103, 100, 99, 101, 103, 105, 104,
                           102, 101, 103, 104, 106, 105])
        atr = _compute_atr(prices, period=14)
        assert atr > 0

    def test_insufficient_data(self):
        assert _compute_atr(np.array([100, 101]), period=14) == 0.0


class TestAvgCorrelation:
    def test_identical_returns(self):
        """Identical series should have correlation near 1."""
        returns = np.random.default_rng(42).normal(0, 0.02, (100, 1))
        matrix = np.column_stack([returns, returns])
        corr = _compute_avg_correlation(matrix)
        assert corr > 0.99

    def test_single_asset(self):
        """Single asset should return 0 correlation."""
        returns = np.random.default_rng(42).normal(0, 0.02, (100, 1))
        assert _compute_avg_correlation(returns) == 0.0


class TestAnnualizedSharpe:
    def test_positive_returns(self):
        returns = np.array([0.001] * 252)
        sharpe = _annualized_sharpe(returns)
        assert sharpe > 0

    def test_zero_returns(self):
        returns = np.zeros(100)
        assert _annualized_sharpe(returns) == 0.0

    def test_short_series(self):
        assert _annualized_sharpe(np.array([0.01])) == 0.0


# ---------------------------------------------------------------------------
# Walk-forward integration tests (synthetic data)
# ---------------------------------------------------------------------------

class TestWalkForwardTest:
    def test_basic_structure(self):
        """Walk-forward result should have expected keys and window count."""
        result = walk_forward_test(
            tickers=["AAPL", "MSFT", "GOOGL"],
            train_days=100,
            test_days=25,
            n_windows=3,
            fetch_fn=_synthetic_fetcher,
        )
        assert "windows" in result
        assert "aggregate" in result
        assert len(result["windows"]) == 3

    def test_window_fields(self):
        """Each window should contain all expected fields."""
        result = walk_forward_test(
            tickers=["AAPL", "MSFT"],
            train_days=80,
            test_days=20,
            n_windows=2,
            fetch_fn=_synthetic_fetcher,
        )
        expected_keys = {
            "window", "train_period", "test_period",
            "strategy_return", "baseline_return", "excess_return",
            "strategy_sharpe", "baseline_sharpe",
            "hursts", "avg_correlation",
        }
        for w in result["windows"]:
            assert expected_keys.issubset(w.keys()), f"Missing keys: {expected_keys - set(w.keys())}"

    def test_aggregate_fields(self):
        """Aggregate should contain summary statistics."""
        result = walk_forward_test(
            tickers=["A", "B"],
            train_days=80,
            test_days=20,
            n_windows=2,
            fetch_fn=_synthetic_fetcher,
        )
        agg = result["aggregate"]
        assert "avg_strategy_return" in agg
        assert "avg_baseline_return" in agg
        assert "strategy_wins" in agg
        assert "information_ratio" in agg
        assert "n_windows" in agg

    def test_single_ticker(self):
        """Walk-forward should work with a single ticker."""
        result = walk_forward_test(
            tickers=["AAPL"],
            train_days=100,
            test_days=25,
            n_windows=2,
            fetch_fn=_synthetic_fetcher,
        )
        assert len(result["windows"]) == 2

    def test_insufficient_data(self):
        """Should return empty windows when data is insufficient."""
        def short_fetcher(tickers, days):
            return {t: np.array([100.0, 101.0]) for t in tickers}

        result = walk_forward_test(
            tickers=["AAPL"],
            train_days=252,
            test_days=63,
            n_windows=4,
            fetch_fn=short_fetcher,
        )
        assert len(result["windows"]) == 0
        assert result["aggregate"]["n_windows"] == 0

    def test_returns_are_finite(self):
        """All return values should be finite numbers."""
        result = walk_forward_test(
            tickers=["A", "B", "C"],
            train_days=100,
            test_days=25,
            n_windows=3,
            fetch_fn=_synthetic_fetcher,
        )
        for w in result["windows"]:
            assert np.isfinite(w["strategy_return"])
            assert np.isfinite(w["baseline_return"])
            assert np.isfinite(w["strategy_sharpe"])
            assert np.isfinite(w["baseline_sharpe"])

    def test_hurst_values_in_range(self):
        """Hurst exponents should be within (0, 1)."""
        result = walk_forward_test(
            tickers=["AAPL", "MSFT"],
            train_days=100,
            test_days=25,
            n_windows=2,
            fetch_fn=_synthetic_fetcher,
        )
        for w in result["windows"]:
            for t, h in w["hursts"].items():
                assert 0.01 <= h <= 0.99, f"Hurst {h} for {t} out of range"


# ---------------------------------------------------------------------------
# CLI integration test (mocked data)
# ---------------------------------------------------------------------------

class _FakeConfig:
    class llm:
        provider = "openai"
        model = "gpt-4o-mini"
        temperature = 0.3
        max_tokens = 2000

    class portfolio:
        db_path = "data/trading_master.db"
        snapshot_dir = "data/snapshots"
        default_cash = 10000.0

    class analysis:
        debate_rounds = 1
        parallel_analysts = True
        cache_ttl_hours = 4

    class risk:
        max_position_pct = 8.0
        max_sector_pct = 20.0
        stop_loss_pct = 8.0
        holding_days = 20
        tail_multiplier = 2.0

    class budget:
        max_cost_per_run = 5.0
        warn_cost = 2.0
        max_tokens_per_run = 500_000

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


@pytest.fixture
def _isolate_db(tmp_path):
    from trading_master.db import Database

    fake_cfg = _FakeConfig()
    fake_cfg.project_root = tmp_path

    db = Database(db_path=tmp_path / "test.db")
    _ = db.conn

    with patch("trading_master.config.get_config", return_value=fake_cfg), \
         patch("trading_master.config._config", fake_cfg), \
         patch("trading_master.config.get_db_path", return_value=tmp_path / "test.db"), \
         patch("trading_master.db.get_db", return_value=db), \
         patch("trading_master.db._db", db):
        yield db

    db.close()


def test_cli_walk_forward(_isolate_db):
    """The walk-forward CLI command should run successfully with mocked data."""
    with patch(
        "trading_master.portfolio.walk_forward._fetch_history",
        side_effect=_synthetic_fetcher,
    ):
        result = runner.invoke(app, [
            "backtest", "walk-forward",
            "--tickers", "AAPL,MSFT",
            "--windows", "2",
            "--train-days", "80",
            "--test-days", "20",
        ])
    assert result.exit_code == 0, result.output
    assert "Walk-Forward" in result.output


def test_cli_backtest_accuracy_no_data(_isolate_db):
    """The accuracy subcommand should handle empty DB gracefully."""
    result = runner.invoke(app, ["backtest", "accuracy"])
    assert result.exit_code == 0, result.output
    assert "No recommendations" in result.output


def test_cli_backtest_help(_isolate_db):
    """The backtest help should show both subcommands."""
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0, result.output
    assert "walk-forward" in result.output
    assert "accuracy" in result.output
