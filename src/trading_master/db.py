"""SQLite database layer with full audit trail."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_db_path
from .models import Action, ActionRecord, ActionSource, Recommendation

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    quantity REAL NOT NULL DEFAULT 0,
    avg_cost REAL NOT NULL DEFAULT 0,
    sector TEXT DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    reasoning TEXT DEFAULT '',
    portfolio_before TEXT DEFAULT '{}',
    portfolio_after TEXT DEFAULT '{}',
    FOREIGN KEY (ticker) REFERENCES positions(ticker)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    analyst_reports TEXT DEFAULT '[]',
    risk_assessment TEXT DEFAULT '{}',
    debate_notes TEXT DEFAULT '',
    timestamp TEXT NOT NULL,
    llm_tokens_used INTEGER DEFAULT 0,
    llm_cost_usd REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    portfolio_json TEXT NOT NULL,
    source TEXT DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_CREATE_SQL)
            # Run schema migrations after initial table creation
            from .db_migrations import run_migrations
            run_migrations(self)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Positions ──────────────────────────────────────────────────

    def get_position(self, ticker: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM positions WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
        return dict(row) if row else None

    def get_all_positions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE quantity > 0 ORDER BY ticker"
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_position(self, ticker: str, quantity: float, avg_cost: float, sector: str = "") -> None:
        self.conn.execute(
            """INSERT INTO positions (ticker, quantity, avg_cost, sector, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                 quantity = excluded.quantity,
                 avg_cost = excluded.avg_cost,
                 sector = CASE WHEN excluded.sector = '' THEN sector ELSE excluded.sector END,
                 updated_at = excluded.updated_at""",
            (ticker.upper(), quantity, avg_cost, sector, datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_cash(self) -> float:
        row = self.conn.execute(
            "SELECT value FROM cache WHERE key = 'cash_balance'"
        ).fetchone()
        if row:
            return float(json.loads(row["value"]))
        from .config import get_config
        return get_config().portfolio.default_cash

    def set_cash(self, amount: float) -> None:
        self.conn.execute(
            """INSERT INTO cache (key, value, expires_at)
               VALUES ('cash_balance', ?, '9999-12-31')
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (json.dumps(amount),),
        )
        self.conn.commit()

    # ── Actions ────────────────────────────────────────────────────

    def log_action(self, record: ActionRecord) -> int:
        cur = self.conn.execute(
            """INSERT INTO actions (ticker, action, quantity, price, timestamp, source, reasoning, portfolio_before, portfolio_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.ticker.upper(),
                record.action.value,
                record.quantity,
                record.price,
                record.timestamp.isoformat(),
                record.source.value,
                record.reasoning,
                json.dumps(record.portfolio_before),
                json.dumps(record.portfolio_after),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def get_actions(self, ticker: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM actions WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM actions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Recommendations ────────────────────────────────────────────

    def save_recommendation(self, rec: Recommendation) -> int:
        cur = self.conn.execute(
            """INSERT INTO recommendations (ticker, action, confidence, summary, analyst_reports, risk_assessment, debate_notes, timestamp, llm_tokens_used, llm_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.ticker.upper(),
                rec.action.value,
                rec.confidence,
                rec.summary,
                json.dumps([r.model_dump() for r in rec.analyst_reports]),
                json.dumps(rec.risk_assessment.model_dump() if rec.risk_assessment else {}),
                rec.debate_notes,
                rec.timestamp.isoformat(),
                rec.llm_tokens_used,
                rec.llm_cost_usd,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def get_recommendations(self, ticker: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM recommendations WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Snapshots ──────────────────────────────────────────────────

    def save_snapshot(self, portfolio_json: dict[str, Any], source: str = "system") -> int:
        cur = self.conn.execute(
            "INSERT INTO snapshots (timestamp, portfolio_json, source) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), json.dumps(portfolio_json), source),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row:
            d = dict(row)
            d["portfolio_json"] = json.loads(d["portfolio_json"])
            return d
        return None

    # ── Cache ──────────────────────────────────────────────────────

    def cache_get(self, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row and datetime.fromisoformat(row["expires_at"]) > datetime.now():
            return json.loads(row["value"])
        return None

    def cache_set(self, key: str, value: Any, ttl_hours: int = 4) -> None:
        expires = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
        self.conn.execute(
            """INSERT INTO cache (key, value, expires_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at""",
            (key, json.dumps(value), expires),
        )
        self.conn.commit()


# Singleton
_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
