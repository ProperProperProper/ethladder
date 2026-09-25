"""Persistent record of optimization runs and current "winner" configs
per (symbol, interval) — the param library. Mirrors the pattern the
user's other bots (e.g. bybit-kst-walkforward-btc) already use: never
lose a result, keep every run so winners can be retested against fresh
data over time and compared against newly-discovered candidates.

Plain sqlite3, not SQLAlchemy — this is a narrow, single-table need, not
the full bot/deal persistence layer (which the project plan deliberately
defers until actually needed).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "optimization_results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    period_days INTEGER NOT NULL,
    tested_at TEXT NOT NULL,
    params_json TEXT NOT NULL,
    in_sample_score REAL,
    out_of_sample_return_quote REAL,
    out_of_sample_return_percent REAL,
    out_of_sample_trade_count INTEGER,
    out_of_sample_win_rate REAL,
    max_drawdown_percent REAL,
    max_funds_required REAL,
    run_kind TEXT NOT NULL,      -- 'search' | 'retest' | 'search_safe' | 'manual'
    is_current_winner INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_symbol_interval ON optimization_results(symbol, interval);
"""

# record() always inserts with is_current_winner=0 regardless of
# run_kind — the ONLY way a row's is_current_winner ever becomes 1 is an
# explicit promote_to_winner() call. get_current_winner/
# get_best_current_winner both filter on is_current_winner=1 alone, so a
# freshly recorded row is invisible to them until something explicitly
# promotes it.

# Added after the initial schema, so upgraded via ALTER TABLE in connect()
# rather than a fresh CREATE TABLE — existing rows (the growing param
# library) are never dropped or rebuilt to pick these up, matching the
# "never wipe the param library" policy used by the user's other bots.
EXTRA_COLUMNS: dict[str, str] = {
    "starting_equity": "REAL",
    "final_equity": "REAL",
    "sharpe_ratio": "REAL",
    "cagr_percent": "REAL",
    "profit_factor": "REAL",
    "gross_profit_quote": "REAL",
    "gross_loss_quote": "REAL",
    "max_win_quote": "REAL",
    "max_loss_quote": "REAL",
    "max_win_percent": "REAL",
    "max_loss_percent": "REAL",
    "average_win_quote": "REAL",
    "average_loss_quote": "REAL",
    "average_win_percent": "REAL",
    "average_loss_percent": "REAL",
    "max_win_streak": "INTEGER",
    "max_loss_streak": "INTEGER",
    "total_fees_quote": "REAL",
    "total_funding_quote": "REAL",
    "liquidation_count": "INTEGER",
    "average_trade_duration_hours": "REAL",
    "average_safety_orders_used": "REAL",
}


@dataclass
class OptimizationRecord:
    symbol: str
    interval: str
    period_days: int
    params: dict
    in_sample_score: float
    out_of_sample_return_quote: float
    out_of_sample_return_percent: float
    out_of_sample_trade_count: int
    out_of_sample_win_rate: float
    max_drawdown_percent: float
    max_funds_required: float
    run_kind: str  # "search" or "retest"
    # Full stat set, matching the manual backtest page ("as much useful
    # info as possible everywhere") — optional so older call sites keep
    # working, but the continuous optimizer always fills these in.
    starting_equity: float = 0.0
    final_equity: float = 0.0
    sharpe_ratio: float = 0.0
    cagr_percent: float = 0.0
    profit_factor: float = 0.0
    gross_profit_quote: float = 0.0
    gross_loss_quote: float = 0.0
    max_win_quote: float = 0.0
    max_loss_quote: float = 0.0
    max_win_percent: float = 0.0
    max_loss_percent: float = 0.0
    average_win_quote: float = 0.0
    average_loss_quote: float = 0.0
    average_win_percent: float = 0.0
    average_loss_percent: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    total_fees_quote: float = 0.0
    total_funding_quote: float = 0.0
    liquidation_count: int = 0
    average_trade_duration_hours: float = 0.0
    average_safety_orders_used: float = 0.0


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(optimization_results)")}
    for column, sql_type in EXTRA_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE optimization_results ADD COLUMN {column} {sql_type}")
    conn.commit()
    return conn


def record(conn: sqlite3.Connection, rec: OptimizationRecord) -> int:
    cur = conn.execute(
        """
        INSERT INTO optimization_results
        (symbol, interval, period_days, tested_at, params_json, in_sample_score,
         out_of_sample_return_quote, out_of_sample_return_percent,
         out_of_sample_trade_count, out_of_sample_win_rate,
         max_drawdown_percent, max_funds_required, run_kind, is_current_winner,
         starting_equity, final_equity, sharpe_ratio, cagr_percent, profit_factor,
         gross_profit_quote, gross_loss_quote, max_win_quote, max_loss_quote,
         max_win_percent, max_loss_percent, average_win_quote, average_loss_quote,
         average_win_percent, average_loss_percent, max_win_streak, max_loss_streak,
         total_fees_quote, total_funding_quote, liquidation_count,
         average_trade_duration_hours, average_safety_orders_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec.symbol, rec.interval, rec.period_days,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(rec.params), rec.in_sample_score,
            rec.out_of_sample_return_quote, rec.out_of_sample_return_percent,
            rec.out_of_sample_trade_count, rec.out_of_sample_win_rate,
            rec.max_drawdown_percent, rec.max_funds_required, rec.run_kind,
            rec.starting_equity, rec.final_equity, rec.sharpe_ratio, rec.cagr_percent,
            rec.profit_factor, rec.gross_profit_quote, rec.gross_loss_quote,
            rec.max_win_quote, rec.max_loss_quote, rec.max_win_percent, rec.max_loss_percent,
            rec.average_win_quote, rec.average_loss_quote, rec.average_win_percent,
            rec.average_loss_percent, rec.max_win_streak, rec.max_loss_streak,
            rec.total_fees_quote, rec.total_funding_quote, rec.liquidation_count,
            rec.average_trade_duration_hours, rec.average_safety_orders_used,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_current_winner(conn: sqlite3.Connection, symbol: str, interval: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM optimization_results WHERE symbol=? AND interval=? AND is_current_winner=1"
        " ORDER BY id DESC LIMIT 1",
        (symbol, interval),
    ).fetchone()


def get_best_current_winner(conn: sqlite3.Connection, symbol: str) -> Optional[sqlite3.Row]:
    """The single best current winner across ALL intervals for `symbol`,
    ranked by in_sample_score (the continuous optimizer's own scoring —
    real-dollar return/drawdown, not a win-rate/streak label) — this is
    what paper (and eventually live) trading should actually run with,
    since it's the config that's currently winning overall, not just
    within one interval's own bucket.

    `liquidation_count=0 OR IS NULL` is a defense-in-depth belt-and-
    suspenders check: promote_to_winner's caller (continuous_optimizer.py)
    is the actual enforcement point (a candidate that liquidated in its
    own test window is never promoted in the first place), but this read
    also refuses to ever hand out a liquidated config even if something
    else ever promoted one by mistake. NULL is allowed through because it
    means a pre-migration row that predates this column entirely, not a
    confirmed-safe or confirmed-liquidated one — excluding NULL outright
    would just make every old row invisible.
    """
    return conn.execute(
        "SELECT * FROM optimization_results WHERE symbol=? AND is_current_winner=1"
        " AND (liquidation_count = 0 OR liquidation_count IS NULL)"
        " ORDER BY in_sample_score DESC LIMIT 1",
        (symbol,),
    ).fetchone()


def promote_to_winner(conn: sqlite3.Connection, symbol: str, interval: str, result_id: int) -> None:
    conn.execute(
        "UPDATE optimization_results SET is_current_winner=0 WHERE symbol=? AND interval=?",
        (symbol, interval),
    )
    conn.execute("UPDATE optimization_results SET is_current_winner=1 WHERE id=?", (result_id,))
    conn.commit()


def demote_winner(conn: sqlite3.Connection, symbol: str, interval: str) -> None:
    """Clears is_current_winner for this (symbol, interval) — used when a
    retest on fresh data shows the current winner now liquidates. Leaves
    no winner in place for this interval until the next cycle finds a
    genuinely safe (zero-liquidation) replacement; paper trading falls
    back to FALLBACK_BOT_DEFAULTS in the meantime rather than keep
    running a config now known to blow up.
    """
    conn.execute(
        "UPDATE optimization_results SET is_current_winner=0 WHERE symbol=? AND interval=?",
        (symbol, interval),
    )
    conn.commit()


def list_winners(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM optimization_results WHERE is_current_winner=1 ORDER BY symbol, interval"
    ).fetchall()


def history(conn: sqlite3.Connection, symbol: str, interval: str, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM optimization_results WHERE symbol=? AND interval=? ORDER BY id DESC LIMIT ?",
        (symbol, interval, limit),
    ).fetchall()


def row_params(row: sqlite3.Row) -> dict:
    return json.loads(row["params_json"])
