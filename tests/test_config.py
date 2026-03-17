"""Tests for configuration loading and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_master.config import (
    AnalysisConfig,
    AppConfig,
    BudgetConfig,
    CircuitBreakerConfig,
    LLMConfig,
    PortfolioConfig,
    RiskConfig,
    load_config,
    get_db_path,
    get_snapshot_dir,
    get_env,
)


# ── Default values ─────────────────────────────────────────────────


class TestConfigDefaults:
    def test_llm_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2000

    def test_portfolio_defaults(self):
        cfg = PortfolioConfig()
        assert cfg.db_path == "data/trading_master.db"
        assert cfg.default_cash == 10000.0

    def test_analysis_defaults(self):
        cfg = AnalysisConfig()
        assert cfg.debate_rounds == 1
        assert cfg.parallel_analysts is True
        assert cfg.cache_ttl_hours == 4

    def test_risk_defaults(self):
        cfg = RiskConfig()
        assert cfg.max_position_pct == 8.0
        assert cfg.max_sector_pct == 20.0
        assert cfg.stop_loss_pct == 8.0
        assert cfg.holding_days == 20
        assert cfg.tail_multiplier == 2.0

    def test_budget_defaults(self):
        cfg = BudgetConfig()
        assert cfg.max_cost_per_run == 5.0
        assert cfg.warn_cost == 2.0
        assert cfg.max_tokens_per_run == 500_000

    def test_circuit_breaker_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.max_drawdown_pct == 15.0

    def test_app_config_all_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.portfolio, PortfolioConfig)
        assert isinstance(cfg.analysis, AnalysisConfig)
        assert isinstance(cfg.risk, RiskConfig)
        assert isinstance(cfg.budget, BudgetConfig)
        assert isinstance(cfg.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(cfg.project_root, Path)


# ── Custom values ──────────────────────────────────────────────────


class TestConfigCustomValues:
    def test_llm_custom(self):
        cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet", temperature=0.7, max_tokens=4000)
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-3-5-sonnet"

    def test_risk_custom(self):
        cfg = RiskConfig(max_position_pct=5.0, holding_days=10, tail_multiplier=3.0)
        assert cfg.max_position_pct == 5.0
        assert cfg.holding_days == 10

    def test_app_config_nested_override(self):
        cfg = AppConfig(llm=LLMConfig(provider="ollama"), risk=RiskConfig(max_position_pct=3.0))
        assert cfg.llm.provider == "ollama"
        assert cfg.risk.max_position_pct == 3.0
        # Non-overridden sections keep defaults
        assert cfg.portfolio.default_cash == 10000.0


# ── load_config ────────────────────────────────────────────────────


class TestLoadConfig:
    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = load_config(config_path=tmp_path / "nonexistent.toml")
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.risk.max_position_pct == 8.0

    def test_loads_from_toml(self, tmp_path):
        toml_content = """
[llm]
provider = "anthropic"
model = "claude-3-5-sonnet"

[risk]
max_position_pct = 5.0
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_path=config_file)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3-5-sonnet"
        assert cfg.risk.max_position_pct == 5.0
        # Non-specified values keep defaults
        assert cfg.llm.temperature == 0.3
        assert cfg.portfolio.default_cash == 10000.0

    def test_partial_toml(self, tmp_path):
        toml_content = """
[budget]
max_cost_per_run = 10.0
"""
        config_file = tmp_path / "partial.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_path=config_file)
        assert cfg.budget.max_cost_per_run == 10.0
        assert cfg.llm.provider == "openai"  # default preserved

    def test_empty_toml(self, tmp_path):
        config_file = tmp_path / "empty.toml"
        config_file.write_text("")

        cfg = load_config(config_path=config_file)
        assert cfg.llm.model == "gpt-4o-mini"


# ── get_db_path / get_snapshot_dir ─────────────────────────────────


class TestPathHelpers:
    def test_get_db_path_creates_parent(self, tmp_path, monkeypatch):
        cfg = AppConfig(project_root=tmp_path)
        monkeypatch.setattr("trading_master.config._config", cfg)
        monkeypatch.setattr("trading_master.config.get_config", lambda: cfg)

        db_path = get_db_path()
        assert db_path.parent.exists()
        assert "trading_master.db" in str(db_path)

    def test_get_snapshot_dir_creates_dir(self, tmp_path, monkeypatch):
        cfg = AppConfig(project_root=tmp_path)
        monkeypatch.setattr("trading_master.config._config", cfg)
        monkeypatch.setattr("trading_master.config.get_config", lambda: cfg)

        snap_dir = get_snapshot_dir()
        assert snap_dir.exists()
        assert snap_dir.is_dir()


# ── get_env ────────────────────────────────────────────────────────


class TestGetEnv:
    def test_returns_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_TM_VAR", "hello")
        assert get_env("TEST_TM_VAR") == "hello"

    def test_returns_default_when_missing(self):
        assert get_env("DEFINITELY_NOT_SET_XYZ", "fallback") == "fallback"

    def test_default_empty_string(self):
        assert get_env("DEFINITELY_NOT_SET_XYZ") == ""
