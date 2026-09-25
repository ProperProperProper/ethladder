"""Read-only JSON endpoint exposing run_everything.py's optimizer task's
live progress (data/optimizer_status.json) plus its current promoted
winners (strategy/optimization_store.py) to the web GUI. This route
never starts, stops, or otherwise controls the optimizer — it only
reads whatever it has already recorded.

Previously the master dashboard's "Backtest Results" tab had NO
connection to any of this at all — it read backtest_summary.json,
itself built from backtest_*.json files that were only ever produced by
a manual single-backtest web form deleted in an earlier cleanup pass
(see CLAUDE.md's "Recent Significant Changes", 2026-09-17: "removed
manual-backtest web UI (/backtest, optimize_cli.py)"). The ONLY page
that ever showed real continuous-optimizer data was /winners (this
same optimization_store.py, via winners.py) — this endpoint is what
lets the master dashboard show the same real data live, without
needing to navigate away.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from symbot_python.strategy.optimization_store import connect, list_winners

logger = logging.getLogger(__name__)

STATUS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "optimizer_status.json"

router = APIRouter()


def _live_status() -> dict:
    if not STATUS_PATH.exists():
        return {"state": "not_started"}
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {"state": "unknown"}


def _current_winners() -> list[dict]:
    conn = None
    try:
        conn = connect()
        winners = []
        for row in list_winners(conn):
            d = dict(row)
            winners.append({
                "interval": d.get("interval"),
                "run_kind": d.get("run_kind"),
                "tested_at": d.get("tested_at"),
                "out_of_sample_return_percent": d.get("out_of_sample_return_percent"),
                "out_of_sample_trade_count": d.get("out_of_sample_trade_count"),
                "out_of_sample_win_rate": d.get("out_of_sample_win_rate"),
                "max_drawdown_percent": d.get("max_drawdown_percent"),
                "sharpe_ratio": d.get("sharpe_ratio"),
                "profit_factor": d.get("profit_factor"),
                "liquidation_count": d.get("liquidation_count"),
                "starting_equity": d.get("starting_equity"),
            })
        return winners
    except Exception:
        logger.exception("Failed to load current winners for /api/optimizer-status.")
        return []
    finally:
        if conn is not None:
            conn.close()


@router.get("/api/optimizer-status")
async def optimizer_status() -> dict:
    status = _live_status()
    status["current_winners"] = _current_winners()
    return status
