#!/usr/bin/env python3
"""
Trading Master Evolution Monitor — Real-time TUI Dashboard
Tails the loop log and renders a split-screen Rich dashboard.

Usage:
    python tm-monitor.py                     # monitor latest log
    python tm-monitor.py logs/loop/xxx.log   # monitor specific log
"""

import sys
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Config ─────────────────────────────────────────────────────────────

PROJECT_DIR = r"D:\trading-master"
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "loop")
MAX_MESSAGES = 30
MAX_TOOLS = 20
MAX_EVENTS = 50
REFRESH_RATE = 4

SPINNER_FRAMES = [
    ">>>    ", " >>>   ", "  >>>  ", "   >>> ",
    "    >>>", "   >>> ", "  >>>  ", " >>>   ",
]


# ── State ──────────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self.tick = 0
        self.iteration = 0
        self.max_iterations = 999
        self.model = "?"
        self.session_id = ""
        self.start_time = datetime.now()
        self.iter_start_time = datetime.now()

        # Current iteration
        self.tool_uses: list[tuple[str, str]] = []
        self.assistant_msgs: list[str] = []
        self.subagent_count = 0
        self.tool_count = 0
        self.rate_limit_ok = True
        self.rate_limit_resets = ""

        # Cumulative
        self.total_cost = 0.0
        self.iter_cost = 0.0
        self.total_tools = 0
        self.total_turns = 0
        self.commits: list[str] = []
        self.errors: list[tuple[int, str]] = []
        self.iterations_done: list[tuple] = []

        # Event log
        self.events: deque = deque(maxlen=MAX_EVENTS)

        # Status
        self.status = "WAITING"
        self.status_detail = ""
        self.last_log_time = ""

        # Judge verdicts (extracted from assistant text)
        self.last_judges: list[str] = []

    def new_iteration(self, num: int) -> None:
        self.iteration = num
        self.iter_start_time = datetime.now()
        self.tool_uses = []
        self.assistant_msgs = []
        self.subagent_count = 0
        self.tool_count = 0
        self.iter_cost = 0.0
        self.rate_limit_ok = True
        self.status = "RUNNING"
        self.status_detail = f"Iteration {num}"
        self.last_judges = []

    def add_event(self, tag: str, msg: str, style: str = "white") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append((ts, tag, msg, style))


state = DashboardState()


# ── Log Parser ─────────────────────────────────────────────────────────

def parse_log_line(line: str) -> None:
    line = line.strip()
    if not line:
        return

    m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)", line)
    if not m:
        return
    ts, content = m.group(1), m.group(2)
    state.last_log_time = ts

    # Iteration header
    m2 = re.match(r"ITERATION (\d+) / (\d+)", content)
    if m2:
        num = int(m2.group(1))
        state.max_iterations = int(m2.group(2))
        state.new_iteration(num)
        state.add_event("ITER", f"Starting iteration {num}", "bold cyan")
        return

    # INIT
    if "[INIT]" in content:
        m3 = re.search(r"session=(\S+)\s+model=(\S+)", content)
        if m3:
            state.session_id = m3.group(1)[:12] + "..."
            state.model = m3.group(2)
        state.add_event("INIT", f"Session started, model={state.model}", "green")
        return

    # ASSISTANT
    if "[ASSISTANT]" in content:
        msg = content.replace("[ASSISTANT]", "").strip()
        state.assistant_msgs.append(msg)
        if len(state.assistant_msgs) > MAX_MESSAGES:
            state.assistant_msgs.pop(0)
        # Try to detect judge verdicts
        for prefix in ("Expert 1", "Expert 2", "Expert 3", "Risk Manager", "Quant", "Engineer"):
            if prefix in msg:
                state.last_judges.append(msg[:100])
                break
        state.add_event("MSG", msg[:80], "white")
        return

    # TOOL
    m_tool = re.match(r"\s*\[TOOL #(\d+)\]\s+(\w+)", content)
    if m_tool:
        num = int(m_tool.group(1))
        name = m_tool.group(2)
        preview = content.split("|", 1)[1].strip() if "|" in content else ""
        if len(preview) > 60:
            preview = preview[:60] + "..."
        state.tool_uses.append((name, preview))
        if len(state.tool_uses) > MAX_TOOLS:
            state.tool_uses.pop(0)
        state.tool_count = num
        state.total_tools += 1
        state.add_event("TOOL", f"#{num} {name}", "magenta")
        return

    # SUBAGENT
    if "[SUBAGENT]" in content:
        state.subagent_count += 1
        return

    # RESULT (success)
    if "[RESULT]" in content and "[RESULT ERROR]" not in content:
        m_r = re.search(
            r"turns=(\d+)\s+cost=\$([0-9.]+)\s+duration=(\d+)ms\s+stop=(\S+)",
            content,
        )
        if m_r:
            turns = int(m_r.group(1))
            cost = float(m_r.group(2))
            duration = int(m_r.group(3))
            stop = m_r.group(4)
            state.iter_cost = cost
            state.total_cost += cost
            state.total_turns += turns
            status = "OK" if stop == "end_turn" else stop
            state.iterations_done.append(
                (state.iteration, cost, duration / 1000, state.tool_count, turns, status)
            )
            state.add_event(
                "DONE",
                f"Iter {state.iteration}: ${cost:.4f}, {turns} turns, {duration/1000:.0f}s",
                "bold green",
            )
        return

    # RESULT ERROR
    if "[RESULT ERROR]" in content:
        err_msg = content.replace("[RESULT ERROR]", "").strip()
        state.errors.append((state.iteration, err_msg[:100]))
        state.status = "ERROR"
        state.add_event("ERR", err_msg[:80], "bold red")
        return

    # RATE LIMIT
    if "[RATE LIMIT]" in content:
        if "OK" in content:
            state.rate_limit_ok = True
        else:
            state.rate_limit_ok = False
            m_rl = re.search(r"resets=(.+)", content)
            state.rate_limit_resets = m_rl.group(1) if m_rl else "?"
            state.add_event("RATE", f"Rate limited! Resets: {state.rate_limit_resets}", "bold red")
        return

    # NEW COMMIT
    if "NEW COMMIT:" in content:
        commit_info = content.replace("NEW COMMIT:", "").strip()
        state.commits.append(commit_info)
        state.add_event("GIT", commit_info, "bold green")
        return

    # FAILURE
    if "FAILURE #" in content:
        state.status = "ERROR"
        state.add_event("FAIL", content[:80], "bold red")
        return

    # Sleeping
    if "Next iteration in" in content or "Sleeping" in content:
        state.status = "SLEEPING"
        state.status_detail = content
        return

    # PROCESS ERROR
    if "PROCESS ERROR:" in content:
        state.status = "ERROR"
        state.errors.append((state.iteration, content[:100]))
        state.add_event("CRASH", content[:80], "bold red on white")
        return


# ── Dashboard Panels ───────────────────────────────────────────────────

def make_header() -> Panel:
    elapsed = datetime.now() - state.start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)

    status_style = {
        "RUNNING": "bold green",
        "SLEEPING": "bold yellow",
        "ERROR": "bold red",
        "WAITING": "bold dim",
    }.get(state.status, "white")

    state.tick += 1
    spinner = SPINNER_FRAMES[state.tick % len(SPINNER_FRAMES)]

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right", ratio=1)

    grid.add_row(
        Text(f" {spinner} TM EVOLUTION MONITOR", style="bold cyan"),
        Text(f"[{state.status}]", style=status_style),
        Text(f"{h:02d}:{m:02d}:{s:02d} elapsed ", style="dim"),
    )

    return Panel(grid, style="bold cyan", box=box.DOUBLE)


def make_stats_panel() -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold cyan", width=14)
    table.add_column("Value", style="white")

    table.add_row("Iteration", f"{state.iteration} / {state.max_iterations}")
    table.add_row("Model", state.model)
    table.add_row("Total Cost", f"${state.total_cost:.4f}")
    table.add_row("Iter Cost", f"${state.iter_cost:.4f}")
    table.add_row("Total Tools", str(state.total_tools))
    table.add_row("Total Turns", str(state.total_turns))
    table.add_row("Commits", str(len(state.commits)))
    table.add_row("Errors", str(len(state.errors)))

    rl_text = (
        Text("OK", style="green")
        if state.rate_limit_ok
        else Text(f"LIMITED ({state.rate_limit_resets})", style="bold red")
    )
    table.add_row("Rate Limit", rl_text)
    table.add_row("Last Log", state.last_log_time or "---")

    # Thinking indicator
    if state.last_log_time:
        try:
            now = datetime.now()
            last = datetime.strptime(state.last_log_time, "%H:%M:%S").replace(
                year=now.year, month=now.month, day=now.day
            )
            gap = (now - last).total_seconds()
            if gap > 10:
                dots = "." * (int(gap / 2) % 4 + 1)
                table.add_row("", Text(f"claude thinking{dots}", style="dim yellow"))
        except ValueError:
            pass

    return Panel(table, title="[bold cyan]Stats", border_style="cyan", box=box.ROUNDED)


def make_judges_panel() -> Panel:
    text = Text()
    if state.last_judges:
        for i, verdict in enumerate(state.last_judges[-3:]):
            label = ["Risk Mgr", "Quant", "Engineer"][i] if i < 3 else f"Judge {i+1}"
            text.append(f"  {label}: ", style="bold yellow")
            text.append(f"{verdict}\n", style="white")
    else:
        text.append("  Waiting for judge verdicts...\n", style="dim")

    return Panel(
        text,
        title="[bold yellow]Judge Panel",
        border_style="yellow",
        box=box.ROUNDED,
    )


def make_tools_panel() -> Panel:
    table = Table(box=None, expand=True, show_header=True, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool", style="magenta", width=12)
    table.add_column("Input", style="dim", ratio=1)

    display_tools = state.tool_uses[-12:]
    start_idx = max(0, len(state.tool_uses) - 12)
    for i, (name, preview) in enumerate(display_tools):
        table.add_row(str(start_idx + i + 1), name, preview)

    subtitle = (
        f"[dim]{state.subagent_count} subagent msgs[/]" if state.subagent_count > 0 else ""
    )
    return Panel(
        table,
        title=f"[bold magenta]Tools ({state.tool_count} this iter)",
        subtitle=subtitle,
        border_style="magenta",
        box=box.ROUNDED,
    )


def make_assistant_panel() -> Panel:
    text = Text()
    msgs = state.assistant_msgs[-8:]
    for msg in msgs:
        display = msg if len(msg) <= 120 else msg[:120] + "..."
        text.append(f"  {display}\n", style="white")

    if not msgs:
        text.append("  Waiting for assistant output...\n", style="dim")

    return Panel(text, title="[bold white]Assistant", border_style="white", box=box.ROUNDED)


def make_commits_panel() -> Panel:
    text = Text()
    recent = state.commits[-10:]
    if not recent:
        text.append("  No commits yet...\n", style="dim")
    for c in recent:
        text.append(f"  {c}\n", style="green")

    return Panel(
        text,
        title=f"[bold green]Git Commits ({len(state.commits)})",
        border_style="green",
        box=box.ROUNDED,
    )


def make_iterations_panel() -> Panel:
    table = Table(box=None, expand=True, show_header=True, padding=(0, 1))
    table.add_column("Iter", style="cyan", width=5)
    table.add_column("Cost", style="yellow", width=8)
    table.add_column("Time", style="dim", width=6)
    table.add_column("Tools", style="magenta", width=6)
    table.add_column("Turns", style="white", width=6)
    table.add_column("Status", width=10)

    recent = state.iterations_done[-8:]
    for num, cost, dur, tools, turns, status in recent:
        st_style = "green" if status == "OK" else "red"
        table.add_row(
            str(num), f"${cost:.3f}", f"{dur:.0f}s",
            str(tools), str(turns), Text(status, style=st_style),
        )

    if not recent:
        table.add_row("--", "--", "--", "--", "--", Text("waiting", style="dim"))

    return Panel(
        table,
        title="[bold yellow]Iteration History",
        border_style="yellow",
        box=box.ROUNDED,
    )


def make_event_log_panel() -> Panel:
    text = Text()
    recent = list(state.events)[-15:]
    for ts, tag, msg, style in recent:
        text.append(f"  {ts} ", style="dim")
        text.append(f"[{tag}] ", style=style)
        text.append(f"{msg}\n", style="dim white")

    if not recent:
        text.append("  Waiting for events...\n", style="dim")

    return Panel(text, title="[bold blue]Event Log", border_style="blue", box=box.ROUNDED)


def make_layout() -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=18),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )

    layout["left"].split_column(
        Layout(name="stats", ratio=1),
        Layout(name="judges", size=7),
        Layout(name="iterations", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="assistant", ratio=1),
        Layout(name="tools", ratio=1),
    )

    layout["footer"].split_row(
        Layout(name="commits", ratio=1),
        Layout(name="events", ratio=2),
    )

    return layout


def render_dashboard() -> Layout:
    layout = make_layout()
    layout["header"].update(make_header())
    layout["stats"].update(make_stats_panel())
    layout["judges"].update(make_judges_panel())
    layout["iterations"].update(make_iterations_panel())
    layout["assistant"].update(make_assistant_panel())
    layout["tools"].update(make_tools_panel())
    layout["commits"].update(make_commits_panel())
    layout["events"].update(make_event_log_panel())
    return layout


# ── Log Tailer ─────────────────────────────────────────────────────────

def find_latest_log() -> Path | None:
    log_dir = Path(LOG_DIR)
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("tm_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def tail_log(path: Path):
    """Yield new lines from a log file (like tail -f).
    Re-opens each cycle for Windows cross-process compatibility."""
    file_pos = 0
    leftover = ""

    while True:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(file_pos)
                raw = f.read()
                new_pos = f.tell()
        except (IOError, OSError):
            yield None
            continue

        if not raw:
            yield None
            continue

        file_pos = new_pos
        raw = leftover + raw
        lines = raw.split("\n")

        leftover = lines[-1]
        for line in lines[:-1]:
            if line.strip():
                yield line

        if not lines[:-1]:
            yield None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    console = Console()

    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_log()

    if not log_path or not log_path.exists():
        console.print("[bold red]No loop log file found![/]")
        console.print(f"[dim]Looked in: {LOG_DIR}[/]")
        console.print("[dim]Start tm-loop.ps1 first, or pass a log path as argument.[/]")
        sys.exit(1)

    console.print(f"[bold cyan]Monitoring:[/] {log_path}")
    console.print("[dim]Press Ctrl+C to exit[/]\n")
    time.sleep(1)

    current_log = log_path
    tailer = tail_log(current_log)

    with Live(render_dashboard(), console=console, refresh_per_second=REFRESH_RATE, screen=True) as live:
        try:
            check_counter = 0
            while True:
                for _ in range(200):
                    line = next(tailer, None)
                    if line is None:
                        break
                    parse_log_line(line)

                check_counter += 1
                if check_counter >= 10:
                    check_counter = 0
                    newest = find_latest_log()
                    if newest and newest != current_log:
                        current_log = newest
                        tailer = tail_log(current_log)
                        state.add_event("LOG", f"Switched to {current_log.name}", "bold yellow")

                live.update(render_dashboard())
                time.sleep(0.5)

        except KeyboardInterrupt:
            pass

    console.print("\n[bold cyan]Monitor stopped.[/]")


if __name__ == "__main__":
    main()
