"""Configuration loading from TOML + environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000


class PortfolioConfig(BaseModel):
    db_path: str = "data/trading_master.db"
    snapshot_dir: str = "data/snapshots"
    default_cash: float = 10000.0


class AnalysisConfig(BaseModel):
    debate_rounds: int = 1
    parallel_analysts: bool = True
    cache_ttl_hours: int = 4


class RiskConfig(BaseModel):
    max_position_pct: float = 8.0
    max_sector_pct: float = 20.0
    stop_loss_pct: float = 8.0
    holding_days: int = 20
    tail_multiplier: float = 2.0


class BudgetConfig(BaseModel):
    max_cost_per_run: float = 5.0
    warn_cost: float = 2.0
    max_tokens_per_run: int = 500_000


class CircuitBreakerConfig(BaseModel):
    max_drawdown_pct: float = 15.0


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    project_root: Path = _PROJECT_ROOT


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config from TOML file, falling back to defaults."""
    path = config_path or _PROJECT_ROOT / "config.toml"
    data: dict[str, Any] = {}
    if path.exists():
        data = toml.load(path)
    return AppConfig(**data)


# Singleton
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_db_path() -> Path:
    cfg = get_config()
    p = cfg.project_root / cfg.portfolio.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_snapshot_dir() -> Path:
    cfg = get_config()
    p = cfg.project_root / cfg.portfolio.snapshot_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)
