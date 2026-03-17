"""Typer CLI package — the main entry point for Trading Master."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..logging_config import setup_logging

app = typer.Typer(name="tm", help="Trading Master — AI-powered trading assistant")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO-level logs on console"),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs on console"),
    debug_file: Optional[Path] = typer.Option(None, "--debug-file", help="Write full DEBUG logs to this file"),
) -> None:
    """Trading Master — AI-powered trading assistant."""
    if debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"
    setup_logging(level=level, debug_file=debug_file)


# Wire sub-apps
from .portfolio import portfolio_app  # noqa: E402
from .action import action_app  # noqa: E402
from .risk import risk_app  # noqa: E402
from .stop_loss import stop_loss_app  # noqa: E402
from .allocation import allocation_app  # noqa: E402
from .analyze import register_commands  # noqa: E402
from .backtest_app import backtest_app  # noqa: E402
from .watchlist import watchlist_app, register_alerts_command  # noqa: E402
from .report import register_report_command  # noqa: E402
from .init import register_init_command  # noqa: E402

app.add_typer(portfolio_app, name="portfolio")
app.add_typer(action_app, name="action")
app.add_typer(risk_app, name="risk")
app.add_typer(stop_loss_app, name="stop-loss")
app.add_typer(allocation_app, name="allocation")
app.add_typer(watchlist_app, name="watchlist")
app.add_typer(backtest_app, name="backtest")
register_commands(app)  # analyze, review, macro are top-level
register_alerts_command(app)  # top-level 'alerts' command
register_report_command(app)  # top-level 'report' command
register_init_command(app)  # top-level 'init' wizard
