"""View current param-library winners and recent history — read-only
window into the continuous optimizer's SQLite store
(strategy/optimization_store.py). This page never runs a search itself;
it only displays whatever scripts/continuous_optimizer.py has already
recorded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from symbot_python.strategy.optimization_store import connect, history, list_winners

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d["params_json"])
    # profit_factor is stored NULL for "no losing trades" (would otherwise
    # be +inf, which Jinja2 has no literal for) — same convention as the
    # manual backtest page's profit_factor_display.
    d["profit_factor_display"] = "∞" if d.get("profit_factor") is None else f"{d['profit_factor']:.2f}"
    return d


@router.get("/winners", response_class=HTMLResponse)
async def winners_page(request: Request):
    # SQLite can raise "database is locked" under write contention from
    # the concurrently-running continuous optimizer — without a
    # try/finally here, that would both leak the connection (never
    # closed) and 500 the whole page instead of failing gracefully.
    winners: list[dict] = []
    histories: dict[str, list[dict]] = {}
    error = None
    conn = None
    try:
        conn = connect()
        winners = [_row_to_dict(r) for r in list_winners(conn)]
        histories = {
            f"{w['symbol']}|{w['interval']}": [_row_to_dict(r) for r in history(conn, w["symbol"], w["interval"], limit=10)]
            for w in winners
        }
    except Exception:
        logger.exception("Failed to load winners page data.")
        error = "Could not load the param library right now — try refreshing."
    finally:
        if conn is not None:
            conn.close()
    return templates.TemplateResponse(
        request, "winners.html", {"winners": winners, "histories": histories, "error": error}
    )
