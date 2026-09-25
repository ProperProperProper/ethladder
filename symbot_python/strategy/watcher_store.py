"""SQLite persistence for scripts/log_watcher.py's alerts — matches the
project's existing pattern (see optimization_store.py) rather than a
plain-text file nobody can query. Kept in its own DB file, separate from
optimization_results.db: these are operational alerts about the running
processes, not backtest/optimization results.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "watcher_alerts.db"
MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def record_alert(conn: sqlite3.Connection, source: str, severity: str, message: str) -> int:
    created_at = datetime.now(MELBOURNE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    cur = conn.execute(
        "INSERT INTO alerts (created_at, source, severity, message) VALUES (?, ?, ?, ?)",
        (created_at, source, severity, message),
    )
    conn.commit()
    return cur.lastrowid


def recent_alerts(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
