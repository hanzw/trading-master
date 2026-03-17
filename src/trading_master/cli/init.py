"""Interactive setup wizard for Trading Master (`tm init`)."""

from __future__ import annotations

import os
from pathlib import Path

import click
import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
_CONFIG_PATH = _PROJECT_ROOT / "config.toml"

# ── Risk profile presets ────────────────────────────────────────────
_RISK_PROFILES: dict[str, dict] = {
    "conservative": {
        "risk": {
            "max_position_pct": 5.0,
            "max_sector_pct": 15.0,
            "stop_loss_pct": 5.0,
        },
        "circuit_breaker": {"max_drawdown_pct": 10.0},
    },
    "balanced": {
        "risk": {
            "max_position_pct": 8.0,
            "max_sector_pct": 20.0,
            "stop_loss_pct": 8.0,
        },
        "circuit_breaker": {"max_drawdown_pct": 15.0},
    },
    "growth": {
        "risk": {
            "max_position_pct": 12.0,
            "max_sector_pct": 30.0,
            "stop_loss_pct": 12.0,
        },
        "circuit_breaker": {"max_drawdown_pct": 25.0},
    },
}

_PROVIDER_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.1",
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def register_init_command(app: typer.Typer) -> None:
    """Register the top-level ``init`` command."""

    @app.command()
    def init() -> None:
        """Interactive setup wizard for Trading Master."""
        console.print(
            Panel(
                "[bold]Welcome to Trading Master![/bold]\n"
                "This wizard will walk you through initial configuration.",
                style="bold cyan",
            )
        )

        # ── Step 1: LLM Provider ────────────────────────────────────
        console.print("\n[bold]Step 1: LLM Provider[/bold]")
        provider: str = typer.prompt(
            "Choose LLM provider",
            default="openai",
            type=click.Choice(["openai", "anthropic", "ollama"]),
        )

        api_key: str | None = None
        env_key = _PROVIDER_ENV_KEYS.get(provider)
        if env_key:
            existing = os.getenv(env_key, "")
            hint = " (leave blank to keep current)" if existing else ""
            api_key = typer.prompt(
                f"Enter your {provider} API key{hint}",
                default="",
                hide_input=True,
                show_default=False,
            )
            if not api_key and existing:
                api_key = existing  # keep what's already set

        # ── Step 2: Risk Profile ────────────────────────────────────
        console.print("\n[bold]Step 2: Risk Profile[/bold]")
        _show_risk_table()
        profile: str = typer.prompt(
            "Risk profile",
            default="balanced",
            type=click.Choice(["conservative", "balanced", "growth"]),
        )

        # ── Step 3: Portfolio Import (optional) ─────────────────────
        console.print("\n[bold]Step 3: Portfolio Import (optional)[/bold]")
        import_file: str = typer.prompt(
            "Path to portfolio CSV/JSON (or 'skip')",
            default="skip",
        )

        # ── Step 4: Starting cash ───────────────────────────────────
        console.print("\n[bold]Step 4: Starting Cash[/bold]")
        cash: float = typer.prompt(
            "Starting cash balance (USD)",
            default=10000.0,
            type=float,
        )

        # ── Apply settings ──────────────────────────────────────────
        _write_env(provider, api_key, env_key)
        _write_config(provider, profile, cash)

        if import_file != "skip":
            _run_import(import_file)

        # ── Summary ─────────────────────────────────────────────────
        console.print()
        summary = Table(title="Configuration Summary", show_header=False)
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="white")
        summary.add_row("LLM provider", provider)
        summary.add_row("Model", _PROVIDER_MODELS.get(provider, provider))
        summary.add_row("Risk profile", profile)
        summary.add_row("Starting cash", f"${cash:,.2f}")
        summary.add_row("Portfolio import", import_file)
        console.print(summary)

        console.print(
            Panel(
                "[bold green]Setup complete![/bold green]\n"
                "Try: [cyan]tm portfolio show[/cyan]  or  [cyan]tm analyze AAPL[/cyan]\n\n"
                "[dim]Re-run [bold]tm init[/bold] any time to update settings.[/dim]",
                style="green",
            )
        )


# ── Helpers ─────────────────────────────────────────────────────────


def _show_risk_table() -> None:
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Profile")
    t.add_column("Max Position %", justify="right")
    t.add_column("Max Sector %", justify="right")
    t.add_column("Stop-Loss %", justify="right")
    t.add_column("Max Drawdown %", justify="right")
    for name, preset in _RISK_PROFILES.items():
        r, cb = preset["risk"], preset["circuit_breaker"]
        t.add_row(
            name,
            f"{r['max_position_pct']}",
            f"{r['max_sector_pct']}",
            f"{r['stop_loss_pct']}",
            f"{cb['max_drawdown_pct']}",
        )
    console.print(t)


def _write_env(provider: str, api_key: str | None, env_key: str | None) -> None:
    """Create or update .env with the API key (idempotent)."""
    if not env_key or not api_key:
        return

    lines: list[str] = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text().splitlines()

    # Replace existing key or append
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{env_key}="):
            lines[i] = f"{env_key}={api_key}"
            found = True
            break
    if not found:
        lines.append(f"{env_key}={api_key}")

    _ENV_PATH.write_text("\n".join(lines) + "\n")
    console.print(f"[dim]Wrote {env_key} to .env[/dim]")


def _write_config(provider: str, profile: str, cash: float) -> None:
    """Update config.toml with provider, risk profile, and cash (idempotent)."""
    data: dict = {}
    if _CONFIG_PATH.exists():
        data = toml.load(_CONFIG_PATH)

    # LLM
    data.setdefault("llm", {})
    data["llm"]["provider"] = provider
    data["llm"]["model"] = _PROVIDER_MODELS.get(provider, data["llm"].get("model", "gpt-4o-mini"))

    # Risk profile
    preset = _RISK_PROFILES[profile]
    data.update({
        "risk": {**data.get("risk", {}), **preset["risk"]},
        "circuit_breaker": {**data.get("circuit_breaker", {}), **preset["circuit_breaker"]},
    })

    # Cash
    data.setdefault("portfolio", {})
    data["portfolio"]["default_cash"] = cash

    _CONFIG_PATH.write_text(toml.dumps(data))
    console.print(f"[dim]Updated config.toml (provider={provider}, profile={profile}, cash={cash})[/dim]")


def _run_import(path_str: str) -> None:
    """Delegate to the portfolio import command."""
    from ..portfolio.tracker import PortfolioTracker

    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        console.print(f"[red]File not found: {p}[/red]")
        return

    try:
        tracker = PortfolioTracker()
        tracker.import_portfolio(str(p))
        console.print(f"[green]Imported portfolio from {p.name}[/green]")
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}[/red]")
