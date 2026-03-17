"""Database schema migration support.

Tracks a schema version in the cache table and applies incremental
migrations so that existing databases can be upgraded without data loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import Database

logger = logging.getLogger(__name__)

_VERSION_CACHE_KEY = "schema_version"

# ── Migration registry ────────────────────────────────────────────────
# Each entry is (version_string, list_of_sql_statements).
# Migrations MUST be ordered by version and MUST be idempotent where
# possible (use IF NOT EXISTS, catch "duplicate column" errors, etc.).

MIGRATIONS: list[tuple[str, list[str]]] = [
    # v0.1.0 — initial schema (handled by CREATE IF NOT EXISTS in _CREATE_SQL)
    # v0.1.2 — watchlist.last_alerted_at (handled inline)
    # v0.1.5 — add sector column to positions (if not already present)
    ("0.1.5", [
        "ALTER TABLE positions ADD COLUMN sector TEXT DEFAULT ''",
    ]),
]


def _version_tuple(v: str) -> tuple[int, ...]:
    """Convert '0.1.5' -> (0, 1, 5) for comparison."""
    return tuple(int(x) for x in v.split("."))


def get_db_version(db: "Database") -> str:
    """Read current schema version from the cache table.

    Returns '0.0.0' if no version has been recorded yet (fresh database).
    """
    result = db.cache_get(_VERSION_CACHE_KEY)
    if result is None:
        return "0.0.0"
    return str(result)


def set_db_version(db: "Database", version: str) -> None:
    """Store schema version in the cache table with a very long TTL."""
    db.cache_set(_VERSION_CACHE_KEY, version, ttl_hours=876_000)


def run_migrations(db: "Database") -> list[str]:
    """Apply any pending migrations. Returns list of applied version strings.

    Idempotent -- safe to run multiple times.  Each migration is attempted
    individually; if a statement fails with a "duplicate column" error it
    is silently skipped (the migration was already partially applied).
    """
    current = get_db_version(db)
    current_tuple = _version_tuple(current)
    applied: list[str] = []

    for version, statements in MIGRATIONS:
        if _version_tuple(version) <= current_tuple:
            continue  # already applied

        logger.info("Applying migration v%s", version)
        for sql in statements:
            try:
                db.conn.execute(sql)
                db.conn.commit()
            except Exception as exc:
                # Common case: "duplicate column name" when re-running
                exc_str = str(exc).lower()
                if "duplicate column" in exc_str or "already exists" in exc_str:
                    logger.debug("Migration v%s: skipped (already applied): %s", version, exc)
                else:
                    logger.warning("Migration v%s failed: %s — %s", version, sql, exc)
                    raise

        set_db_version(db, version)
        applied.append(version)
        logger.info("Migration v%s applied successfully", version)

    return applied
