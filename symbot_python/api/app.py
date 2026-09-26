"""Minimal FastAPI web GUI. Manual one-off backtesting was removed —
the continuous optimizer (scripts/continuous_optimizer.py) already
searches and records everything to the SQLite param library
(strategy/optimization_store.py) automatically; a manual form duplicated
that with no real benefit. This module now just hosts the paper trading
and winners pages, plus fetch_klines, which continuous_optimizer.py
itself imports from here. Run with:

    uvicorn symbot_python.api.app:app --reload

Then open http://127.0.0.1:8000/paper
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pybit.unified_trading import HTTP

from symbot_python.api import console as console_module
from symbot_python.api import inbox_candidates as inbox_candidates_module
from symbot_python.api import unified_trading as unified_trading_module
from symbot_python.api import optimizer_status as optimizer_status_module
from symbot_python.api import paper as paper_module
from symbot_python.api import winners as winners_module
from symbot_python.logging_setup import configure_logging, log_file_unless_testing
from symbot_python.signals.candles import is_native_interval, resample_candles
from symbot_python.strategy.optimizer_control import request_optimizer_run

# Must happen before any module-level logger is used anywhere in this
# process (paper trading's engine-crash logging in particular) — without
# this, uvicorn never calls logging.basicConfig() itself for OUR loggers,
# so an error logged from strategy/dca_bot_manager.py had no timestamp at
# all, just level/name/message. log_file writes directly via
# logging.FileHandler rather than relying on shell/launchd stdout
# redirection — see logging_setup.py's docstring for why that matters.
configure_logging(log_file=log_file_unless_testing(
    Path(__file__).resolve().parent.parent.parent / "logs" / "uvicorn.log"
))

# Locked, not configurable — same policy as the 14-day period cap and the
# real-balance requirement. This whole app trades one instrument only.
# See TESTING_POLICY.md / [[feedback_non_negotiables]].
CATEGORY = "linear"  # perpetual futures, matching bybit_client.py/paper_client.py — never "spot"

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A bot restart must always cause a fresh optimization. The optimizer
    # consumes this local request within one second even when it is between
    # its ordinary hourly cycles; no privileged launchctl control is needed.
    request_optimizer_run()
    yield
    await asyncio.gather(
        paper_module.shutdown_manager(),
        unified_trading_module.shutdown_manager(),
        return_exceptions=True
    )


app = FastAPI(title="EthLadder", lifespan=lifespan)
app.include_router(paper_module.router)
app.include_router(unified_trading_module.router)
app.include_router(winners_module.router)
app.include_router(optimizer_status_module.router)
app.include_router(console_module.router)
app.include_router(inbox_candidates_module.router)


@app.middleware("http")
async def no_cache_headers(request: Request, call_next):
    """Every page here shows live data (auto-optimizer progress, paper
    deals, param-library winners) and several auto-refresh on a timer —
    without an explicit no-store, a browser can serve a cached copy on
    that "refresh" instead of actually hitting the server, so the page
    looks frozen even though the backend data is genuinely moving.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serve new master dashboard
    master_dashboard = Path(__file__).resolve().parent.parent.parent / "ethladder_master_dashboard.html"
    return HTMLResponse(master_dashboard.read_text())


# Real bug this fixes: the master dashboard's JS does fetch('omlx_metrics_live.json')
# etc. (relative paths) for its live data — but with no dedicated route
# for any of them, EVERY one of those requests fell through to the
# catch_all route below, which returns the dashboard's own HTML for any
# unmatched path. fetchJSON()'s resp.json() then threw parsing HTML as
# JSON, was silently swallowed by its own try/catch, and returned null
# — every single card on the dashboard showed "--" forever, regardless
# of whether run_everything.py's export tasks were producing real data
# (they were). Explicit routes, registered before catch_all (FastAPI
# matches in registration order), one per file the dashboard actually
# fetches — not a generic {filename}.json pattern, to keep this from
# ever serving an arbitrary file by name.
DASHBOARD_JSON_FILES = (
    "omlx_metrics_live.json",
    "paper_trading_summary.json",
    "positions.json",
    "reverse_trading_summary.json",
    "system_metrics.json",
)


def _serve_dashboard_json(filename: str) -> FileResponse:
    path = Path(__file__).resolve().parent.parent.parent / filename
    if not path.exists():
        # Not created yet (fresh install, before the first export cycle)
        # — 404 is handled gracefully by the dashboard's own fetchJSON
        # (resp.ok check), same as any other transient fetch failure.
        raise HTTPException(status_code=404, detail=f"{filename} not generated yet")
    return FileResponse(path, media_type="application/json")


for _json_filename in DASHBOARD_JSON_FILES:
    app.add_api_route(
        f"/{_json_filename}",
        lambda filename=_json_filename: _serve_dashboard_json(filename),
        methods=["GET"],
    )


@app.post("/api/analyze-funds-utilization")
async def analyze_funds_utilization():
    """Run automated analysis on funds utilization percentages (78-100%).
    Tests each percentage and returns ranking by profit factor."""
    from symbot_python.system.funds_utilization_analyzer import analyze_all_utilizations
    result = await analyze_all_utilizations()
    return result


@app.get("/api/credentials-status")
async def credentials_status():
    """Check if API credentials are stored in Keychain."""
    from symbot_python.exchange.keychain_manager import check_credentials_exist
    exists = check_credentials_exist()
    return {"has_credentials": exists}


@app.post("/api/credentials-save")
async def credentials_save(request: Request):
    """Save API credentials to Keychain."""
    from symbot_python.exchange.keychain_manager import save_credentials
    try:
        body = await request.json()
        api_key = body.get("api_key", "").strip()
        api_secret = body.get("api_secret", "").strip()
        result = save_credentials(api_key, api_secret)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/credentials-clear")
async def credentials_clear():
    """Remove API credentials from Keychain."""
    from symbot_python.exchange.keychain_manager import clear_credentials
    result = clear_credentials()
    return result


@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str, request: Request):
    # Serve master dashboard for all unknown FRONTEND routes (/backtest,
    # /analytics, etc. — client-side-routed paths with no dedicated
    # FastAPI handler). /api/* is excluded: a genuinely undefined API
    # endpoint must 404, not silently return 200 with dashboard HTML —
    # without this, test_external_telemetry_export_is_removed's whole
    # premise (that /api/telemetry was actually removed, not just
    # unreachable) could never be verified, since EVERY path under /api/
    # would always return 200 regardless of whether a route exists.
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    master_dashboard = Path(__file__).resolve().parent.parent.parent / "ethladder_master_dashboard.html"
    return HTMLResponse(master_dashboard.read_text())


_public_session = HTTP()  # unauthenticated: public market data only, never trades

BYBIT_PAGE_SIZE = 1000
# 200 pages of 1000 1-minute candles ~= 139 days of underlying data, which
# comfortably covers custom-interval (e.g. 29m/39m/49m/59m) aggregation
# requests spanning well past two weeks, while still bounding how many
# sequential requests one fetch can trigger.
MAX_PAGES = 200


async def _fetch_native_klines(symbol: str, interval: str, total_needed: int) -> list[list[float]]:
    """Paginate past Bybit's 1000-candles-per-call cap by walking `end`
    backward in time. Returns oldest-first, trimmed to the most recent
    `total_needed` candles.
    """
    all_rows: list[list] = []
    end_ms: int | None = None
    pages = 0

    while len(all_rows) < total_needed and pages < MAX_PAGES:
        kwargs: dict = {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "limit": BYBIT_PAGE_SIZE,
        }
        if end_ms is not None:
            kwargs["end"] = end_ms

        def call(kwargs=kwargs) -> dict:
            return _public_session.get_kline(**kwargs)

        response = await asyncio.to_thread(call)
        if response.get("retCode") != 0:
            raise ValueError(response.get("retMsg", "Bybit API error"))
        rows = response["result"]["list"]  # newest-first
        pages += 1
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        end_ms = oldest_ts - 1
        if len(rows) < BYBIT_PAGE_SIZE:
            break  # exchange has no more history before this point

    candles = [
        [float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
        for r in all_rows
    ]
    candles.reverse()  # oldest-first
    if len(candles) > total_needed:
        candles = candles[-total_needed:]
    return candles


async def fetch_klines(symbol: str, interval: str, limit: int) -> list[list[float]]:
    """Returns `limit` candles at `interval`, oldest-first.

    `interval` may be a native Bybit interval ("1","5","60","D",...) or
    an arbitrary custom minute width (e.g. "29", "39") — anything not in
    BYBIT_NATIVE_INTERVALS is built by fetching enough 1-minute candles
    and aggregating them (see signals/candles.py). This transparently
    paginates past Bybit's 1000-candle-per-request cap either way, so a
    request for e.g. 2000 hourly candles (~83 days) or 500 29-minute
    candles (~10 days, needing 14500 1-minute candles under the hood)
    both just work. Used by scripts/continuous_optimizer.py.
    """
    if is_native_interval(interval):
        return await _fetch_native_klines(symbol, interval, limit)

    try:
        bucket_minutes = int(interval)
    except ValueError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc
    if bucket_minutes <= 0:
        raise ValueError(f"Unsupported interval: {interval}")

    # +2 buckets of slack: the aggregator only emits a bucket once at
    # least one 1m candle has landed in it, and we trim to `limit` after.
    one_minute_needed = (limit + 2) * bucket_minutes
    one_minute_candles = await _fetch_native_klines(symbol, "1", one_minute_needed)
    aggregated = resample_candles(one_minute_candles, bucket_minutes)
    return aggregated[-limit:] if len(aggregated) > limit else aggregated
