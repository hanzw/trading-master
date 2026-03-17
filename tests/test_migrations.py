"""Tests for database migration logic."""

import sqlite3
from pathlib import Path

import pytest

from trading_master.db import Database
from trading_master.db_migrations import (
    MIGRATIONS,
    get_db_version,
    set_db_version,
    run_migrations,
    _version_tuple,
)


@pytest.fixture
def db(tmp_path):
    """Create a temporary database (triggers table creation + migrations)."""
    db_path = tmp_path / "test_mig.db"
    return Database(db_path)


# ── Version helpers ───────────────────────────────────────────────────


def test_version_tuple():
    assert _version_tuple("0.0.0") == (0, 0, 0)
    assert _version_tuple("0.1.5") == (0, 1, 5)
    assert _version_tuple("1.2.3") == (1, 2, 3)
    assert _version_tuple("0.1.5") > _version_tuple("0.1.0")


# ── get/set version ──────────────────────────────────────────────────


def test_get_version_fresh_db(tmp_path):
    """A brand-new DB (before conn is accessed) should report 0.0.0 initially,
    but after conn access migrations run and set the version."""
    db_path = tmp_path / "fresh.db"
    d = Database(db_path)
    # Accessing .conn triggers table creation + migrations
    _ = d.conn
    # After migrations, version should be the latest migration version
    # (or 0.0.0 if all migrations are skipped because schema already has the columns)
    version = get_db_version(d)
    # The current _CREATE_SQL already includes `sector TEXT DEFAULT ''`
    # so the 0.1.5 ALTER TABLE will fail with "duplicate column" and be skipped,
    # but set_db_version is still called after the migration block.
    # Since the column already exists, the migration is "applied" (version set).
    assert version == "0.1.5"


def test_set_and_get_version(db):
    set_db_version(db, "1.2.3")
    assert get_db_version(db) == "1.2.3"


def test_set_version_overwrites(db):
    set_db_version(db, "0.1.0")
    set_db_version(db, "0.2.0")
    assert get_db_version(db) == "0.2.0"


# ── Migration execution ──────────────────────────────────────────────


def test_migrations_are_idempotent(db):
    """Running migrations twice should not raise errors."""
    # First run happens in db fixture (via conn property).
    # Run again explicitly:
    applied = run_migrations(db)
    # All migrations already applied, nothing new
    assert applied == []


def test_migrations_skip_already_applied(db):
    """If version is already at latest, no migrations run."""
    latest_version = MIGRATIONS[-1][0] if MIGRATIONS else "0.0.0"
    set_db_version(db, latest_version)
    applied = run_migrations(db)
    assert applied == []


def test_migrations_apply_pending(tmp_path):
    """Simulate an older DB that needs migrations."""
    db_path = tmp_path / "old.db"
    # Create a DB manually WITHOUT the sector column to simulate old schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            quantity REAL NOT NULL DEFAULT 0,
            avg_cost REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)
    # Create other required tables
    conn.execute("""
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
            portfolio_after TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
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
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            portfolio_json TEXT NOT NULL,
            source TEXT DEFAULT 'system'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # Now open via Database — _CREATE_SQL will run (CREATE IF NOT EXISTS is safe)
    # and then run_migrations should apply the 0.1.5 ALTER TABLE
    d = Database(db_path)
    _ = d.conn  # triggers creation + migration

    # Verify the sector column now exists
    row = d.conn.execute("PRAGMA table_info(positions)").fetchall()
    col_names = [r[1] for r in row]
    assert "sector" in col_names

    # Verify version was set
    assert get_db_version(d) == "0.1.5"


def test_version_tracking_across_multiple_migrations(tmp_path):
    """Verify that version advances correctly through migrations."""
    db_path = tmp_path / "multi.db"
    d = Database(db_path)
    _ = d.conn

    # Should be at latest migration version
    if MIGRATIONS:
        assert get_db_version(d) == MIGRATIONS[-1][0]
    else:
        assert get_db_version(d) == "0.0.0"


def test_positions_sector_column_exists(db):
    """The sector column should exist after migrations."""
    row = db.conn.execute("PRAGMA table_info(positions)").fetchall()
    col_names = [r[1] for r in row]
    assert "sector" in col_names


def test_sector_column_functional(db):
    """Can insert and read sector data after migration."""
    db.upsert_position("AAPL", 10, 150.0, "Technology")
    pos = db.get_position("AAPL")
    assert pos["sector"] == "Technology"
