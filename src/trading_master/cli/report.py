"""CLI command for daily health report generation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def register_report_command(app: typer.Typer) -> None:
    """Register the top-level `tm report` command."""

    @app.command("report")
    def report(
        cron: bool = typer.Option(
            False, "--cron", help="Plain text output for piping to email/logs (no Rich formatting)"
        ),
        output_dir: Path = typer.Option(
            None, "--output-dir", "-o", help="Directory to save the report file"
        ),
    ) -> None:
        """Generate and display a daily portfolio health report."""
        from ..output.daily_report import generate_daily_report, generate_cron_report

        if cron:
            # Plain text, no Rich — suitable for >> redirection
            text = generate_cron_report()
            print(text)
        else:
            # Save to file and display with Rich
            if output_dir is None:
                output_dir = Path("data/reports")

            filepath = generate_daily_report(output_dir=output_dir)
            report_text = filepath.read_text(encoding="utf-8")
            console.print(report_text)
            console.print(f"\n[dim]Report saved to {filepath}[/dim]")
