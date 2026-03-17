"""Centralized logging configuration with Rich console output."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "WARNING",
    debug_file: Path | None = None,
) -> None:
    """Configure logging with Rich console handler + optional debug file.

    The console handler uses *level* (default ``WARNING``) so the terminal
    stays quiet during normal operation.  When *debug_file* is provided, a
    ``FileHandler`` at ``DEBUG`` level is attached to capture everything.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers decide what to show

    # Remove any existing handlers (avoid duplicates on repeated calls)
    root.handlers.clear()

    # ── Rich console handler ──────────────────────────────────────────
    console_handler = RichHandler(
        level=getattr(logging, level.upper(), logging.WARNING),
        console=None,  # use default stderr console
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(getattr(logging, level.upper(), logging.WARNING))
    root.addHandler(console_handler)

    # ── Optional file handler (DEBUG everything) ──────────────────────
    if debug_file is not None:
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(debug_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)


class AgentTimer:
    """Context manager to time and log agent execution.

    Usage::

        with AgentTimer("fundamental"):
            await analyze_fundamental(state)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = logging.getLogger(f"trading_master.timer.{name}")
        self._start: float = 0.0

    def __enter__(self) -> "AgentTimer":
        self._start = time.perf_counter()
        self._logger.debug("Agent [%s] started", self.name)
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.perf_counter() - self._start
        self._logger.info(
            "Agent [%s] finished in %.2fs", self.name, elapsed
        )
