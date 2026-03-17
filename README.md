# Trading Master

Multi-agent AI portfolio management system with quantitative risk controls.

## Features

- **Multi-agent analysis**: 3 AI analysts (fundamental, technical, sentiment) debate and synthesize recommendations
- **Quantitative risk**: ATR-based position sizing, correlation checks, portfolio CVaR, drawdown circuit breaker
- **Regime awareness**: Macro indicators (VIX, yield curve, S&P 500 vs SMA200) dynamically adjust position sizes and allocation targets
- **Portfolio tracking**: Import from CSV/JSON/text, Robinhood sync, full action audit trail
- **Asset allocation**: Preset models (balanced/growth/conservative) with regime-conditional targets and rebalance suggestions
- **Overlap detection**: Flags duplicate exposures (SPY/VOO, QQQ/QQQM) and suggests consolidation
- **Tax awareness**: Estimated capital gains and holding period in rebalance suggestions
- **Dividend income**: Per-position and portfolio-level dividend yield breakdown
- **Backtesting**: Track recommendation accuracy, per-agent hit rates, calibration analysis
- **Cost budgeting**: Per-run token and dollar limits with warnings before expensive portfolio scans
- **25+ CLI commands**: analyze, portfolio, risk, stop-loss, allocation, macro, backtest, watchlist, alerts

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env          # add your LLM API key
tm portfolio import sample.csv
tm portfolio show
tm analyze AAPL
tm portfolio health
```

## Configuration

Copy `.env.example` to `.env` and set your API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
Edit `config.toml` for risk limits, allocation preferences, LLM model, and budget caps.

Optional integrations (set keys in `.env`):
- **Robinhood** — `tm portfolio sync` pulls live positions
- **Reddit** — sentiment agent reads relevant subreddits

## CLI Reference

| Command Group | Subcommands | Description |
|---|---|---|
| `tm analyze` | `--portfolio` | Run full AI pipeline on a ticker or all holdings |
| `tm review` | `--ticker`, `--limit` | Show pending/recent recommendations |
| `tm backtest` | `--ticker` | Hit rates, per-agent accuracy, calibration |
| `tm macro` | | VIX, yields, regime detection |
| `tm portfolio` | `show`, `import`, `update`, `sync`, `history`, `income`, `health` | Portfolio management and health checks |
| `tm action` | `buy`, `sell` | Manual trade logging |
| `tm risk` | `dashboard`, `correlation`, `sizing` | Sharpe, Sortino, VaR, CVaR, beta, max drawdown |
| `tm stop-loss` | `show`, `set`, `check`, `auto` | Stop-loss management |
| `tm allocation` | `show`, `rebalance` | Allocation vs target model, rebalance suggestions |

## Architecture

```
Data Collection (yfinance, macro, sentiment)
        │
        ▼
┌───────────────────────┐
│  3 Analyst Agents     │  fundamental, technical, sentiment (parallel)
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Debate / Moderator   │  agents challenge each other's views
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  LLM Risk Assessment  │  narrative risk review
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Quantitative Risk    │  ATR sizing, correlation, CVaR
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Synthesis + Circuit  │  final recommendation, drawdown check
│  Breaker              │
└───────────────────────┘
```

The pipeline is built on LangGraph. LLM calls go through a unified adapter supporting OpenAI, Anthropic, and Ollama.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v            # 324+ tests
```

### Project Layout

```
src/trading_master/
├── agents/          # LLM analysts, debate graph, caching
├── cli/             # Typer commands
├── data/            # Market data, fundamentals, macro, sentiment
├── portfolio/       # Tracker, allocation, risk metrics, backtest
├── output/          # Rich report formatting
├── models.py        # Pydantic domain models
├── config.py        # TOML config loader
├── db.py            # SQLite persistence
└── budget.py        # Cost tracking
```

## License

MIT
