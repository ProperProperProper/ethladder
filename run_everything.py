#!/usr/bin/env python3
"""ONE script — the entire ETH Ladder system. One process, one PID.

Every piece that used to be a separate script now lives directly in this
file as an asyncio background task, sharing one FastAPI/uvicorn process:

  - Web server + dashboard   (FastAPI app, symbot_python/api/app.py)
  - Trading bot (paper)      (DCABotManager, driven by the FastAPI app's
                               paper-trading router — already in-process
                               there, so "the bot" is just the app itself)
  - Forward tester           (was run_forward_omlx_tester.py)
  - Continuous optimizer     (was scripts/continuous_optimizer.py)
  - ML trainer               (symbot_python/ml/continuous_trainer.py)
  - Data exporter            (was continuous_data_exporter.py)
  - OMLX metrics exporter    (was export_live_metrics.py)
  - SQLite saver             (was continuous_sqlite_saver.py)
  - Log watcher              (was scripts/log_watcher.py)
  - Resource monitor         (was export_system_metrics.py)
  - Data pruner              (was scheduled_pruner.py / data_pruner.py)

Nothing here reimplements the *strategy/exchange/ML logic* that already
lives in the symbot_python package — dca_bot.py, backtest.py, optimize.py,
forward_omlx_tester.py, the OMLX modules, etc. are real library code and
stay exactly where they are, imported normally. What got inlined is the
former *entry-point glue* — the argparse/while-True/signal-handling
wrapper each old top-level script had around that library code — since
that glue is exactly what "a separate script" meant, and this file
replaces all of it.

Two structural things had to be handled deliberately, not just pasted in:

- CPU-bound work never runs directly on the shared event loop. The
  optimizer's walk-forward search and the forward tester's parameter-
  combination testing are both synchronous, CPU-bound Python with no
  internal `await` points — each runs via asyncio.to_thread() wrapping
  its own private asyncio.run() call, isolated on a worker thread.
  Without this, uvicorn's own ASGI startup handshake never completes and
  the port never opens (observed directly while building this).
- Only one FileHandler is attached to the root logger (see the comment
  right after configure_logging() below) — log_watcher's own alerts are
  never allowed to write back into the file it's tailing.

=============================================================================
WHAT THIS BOT ACTUALLY DOES (the trading strategy, not the plumbing)
=============================================================================

This is a DCA (dollar-cost-averaging) ladder bot for ETHUSDT perpetual
futures on Bybit, 9-11x leverage. The core idea: open a small position,
and if price moves against it, add to the position at progressively
lower (long) or higher (short) prices — a "safety order" ladder — so the
AVERAGE entry price improves as the position averages in, letting a
smaller bounce close the whole ladder in profit than would be needed to
recover the very first, worst-priced entry alone. Take-profit is a small,
frequent target (not a big rare one) — this strategy makes money on
volume of small wins, not on catching big moves.

The two things that make this MORE than a bare DCA bot — OMLX — are:

1. **omlx_bounce_analyzer.py / dip_analysis_service.py**: when the
   position is drawing down, instead of blindly following the pre-built
   safety-order ladder, this scores the CURRENT dip across 10 dimensions
   (volume, price action, momentum, volatility, microstructure, time of
   day, BTC correlation, learned pattern match, technical setup, risk/
   liquidation buffer) into a 0-95% "bounce probability". High
   probability -> add safety order (aggressively, if very high).
   Low probability -> bail out rather than keep averaging into a real
   trend reversal, not just a dip. This is the layer that turns "always
   average down" into "average down only when the data says it's
   actually a bounce setup, not a falling knife".
2. **The ML layer** (`symbot_python/ml/`): every closed trade — win or
   loss — becomes training data. `outcome_predictor.py` (XGBoost) learns
   which feature combinations actually predict a win; `rl_decision_
   optimizer.py` (Q-learning) learns which ACTION (add safety / wait /
   bail) actually paid off in which market state, from real outcomes
   rather than a fixed rulebook. `omlx_ml_advisor.py` blends this with
   the bounce analyzer's own score before the engine acts on it.

Everything else in this file exists to keep that decision loop fed with
real, current data and to keep testing/improving it safely:
- `optimizer` finds the DCA ladder parameters (step %, leverage, take-
  profit, stop-loss, trailing, stop-and-reverse) that actually performed
  best on the last real 14 days, walk-forward validated so a config is
  never trusted on the same data it was chosen on.
- `forward_tester` runs the whole decision loop against real market data
  continuously, generating the trade outcomes `ml_trainer` learns from —
  this is the ONLY source of training data; there is no synthetic data
  path (see FATAL asserts in `load_klines_data`).
- `ml_trainer` is the feedback loop closing itself: better data ->
  better models -> better OMLX decisions -> hopefully-better trades ->
  more/better data.

If you're modifying trading logic and only look at run_everything.py,
you're only seeing the scheduling — the actual decisions are made in
`symbot_python/strategy/dca_bot.py` (the tick loop), `symbot_python/
exchange/dip_analysis_service.py` + `omlx_bounce_analyzer.py` (the "is
this a real bounce" scoring), and `symbot_python/exchange/omlx_ml_advisor.py`
(where ML predictions and the bounce score get combined into one action).

=============================================================================

Run with:
    python3 run_everything.py

Or as a launchd service: scripts/launchd/com.ethladder.unified.plist.
Ctrl+C (or SIGTERM) stops everything together.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import uvicorn
from pybit.unified_trading import HTTP

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from symbot_python.logging_setup import configure_logging, log_file_unless_testing  # noqa: E402
from symbot_python.system.cpu_limiter import get_cpu_limiter  # noqa: E402
from symbot_python.system.resource_monitor import SystemResourceMonitor  # noqa: E402
from symbot_python.exchange.keychain import fetch_real_balance  # noqa: E402
from symbot_python.exchange.utils import call_with_timeout  # noqa: E402
from symbot_python.signals.candles import DEFAULT_BACKTEST_DAYS, bars_for_days  # noqa: E402
from symbot_python.strategy.backtest import FIXED_TAKE_PROFIT_PERCENT, BacktestConfig  # noqa: E402
from symbot_python.strategy.leverage_policy import MAX_LEVERAGE, MIN_LEVERAGE  # noqa: E402
from symbot_python.strategy.funds_utilization_policy import (  # noqa: E402
    MAX_FUNDS_UTILIZATION_PERCENT,
    MIN_FUNDS_UTILIZATION_PERCENT,
)
from symbot_python.strategy.optimize import (  # noqa: E402
    ParamGrid,
    score_lowest_loss_highest_pnl,
    walk_forward,
    walk_forward_fixed,
)
from symbot_python.strategy.optimizer_control import (  # noqa: E402
    consume_optimizer_run_request,
    request_optimizer_run,
)
from symbot_python.strategy.optimization_store import (  # noqa: E402
    OptimizationRecord,
    connect as connect_optimizer_store,
    demote_winner,
    get_best_current_winner,
    get_current_winner,
    promote_to_winner,
    record as record_optimization,
    row_params,
)
from symbot_python.strategy.forward_omlx_tester import run_forward_tester  # noqa: E402
from symbot_python.strategy.watcher_store import connect as connect_watcher_db, record_alert  # noqa: E402
from symbot_python.ml.continuous_trainer import ContinuousMLTrainer  # noqa: E402
from symbot_python.exchange import inbox_pattern_extractor  # noqa: E402

# The FastAPI app already contains: web server routes, the master
# dashboard, and the paper-trading bot manager (symbot_python/api/paper.py).
# Importing it here — rather than shelling out to `uvicorn <module>` — is
# what makes "the bot" and "the web server" the same process as everything
# else in this file. fetch_klines is the same real-BYBIT-only kline fetch
# continuous_optimizer.py and the forward tester both used.
from symbot_python.api.app import app as fastapi_app, fetch_klines  # noqa: E402
from symbot_python.strategy.models import DealStatus  # noqa: E402
from symbot_python.api import unified_trading as trading_module  # noqa: E402

configure_logging(log_file=log_file_unless_testing(REPO_ROOT / "logs" / "run_everything.log"))
log = logging.getLogger("run_everything")

# configure_logging() is designed to be safe to call repeatedly ACROSS
# SEPARATE PROCESSES (each of the scripts this file replaces used to call
# it with its own log_file, each in its own process with its own root
# logger). This file is the only entry point left, so it's also the only
# thing that should ever attach a FileHandler to the root logger —
# guarded here defensively in case anything imported above still does.
for _extra_handler in list(logging.getLogger().handlers):
    if isinstance(_extra_handler, logging.FileHandler):
        _this_log = (REPO_ROOT / "logs" / "run_everything.log").resolve()
        if Path(_extra_handler.baseFilename).resolve() != _this_log:
            logging.getLogger().removeHandler(_extra_handler)
            _extra_handler.close()

# Dedicated logger for the optimizer's own [optimizer]/[<interval>]-tagged
# lines, ALSO written to a small separate file (logs/optimizer_activity.log)
# in addition to the combined run_everything.log (propagate=True, the
# default, sends every record up to the root logger's own handler too —
# this is purely additive, not a replacement).
#
# Real bug this fixes: console.py's "optimizer" tab used to filter a
# fixed-size tail of the combined log for these lines — but this process
# writes an enormous volume of OTHER logging (OMLX dip-analysis chatter
# during forward-test replay alone runs to tens of MB/minute), so even
# an 8 MiB filtered search window could cover well under 10 seconds of
# real time, nowhere near enough to reliably contain the optimizer's own
# genuinely sparse lines (one every few minutes) even while it's working
# perfectly. A small dedicated file sidesteps the volume mismatch
# entirely instead of trying to out-search it.
#
# log_file_unless_testing() guard (same one configure_logging() above
# uses): without it, importing this module under pytest — which the
# test suite does, including tests that deliberately exercise the
# optimizer's failure/logging paths — attaches a real FileHandler on
# the ACTUAL logs/optimizer_activity.log every single test run,
# silently writing test-triggered tracebacks into it. Caught directly:
# a "RuntimeError: simulated total outage" from
# test_continuous_optimizer_resilience.py ended up sitting in the real
# production log file after nothing more than running the test suite.
optimizer_log = logging.getLogger("optimizer_activity")
_optimizer_log_path = log_file_unless_testing(REPO_ROOT / "logs" / "optimizer_activity.log")
if _optimizer_log_path is not None and not any(
    isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == _optimizer_log_path.resolve()
    for h in optimizer_log.handlers
):
    _optimizer_log_handler = logging.FileHandler(_optimizer_log_path)
    _optimizer_log_handler.setFormatter(logging.getLogger().handlers[0].formatter if logging.getLogger().handlers else None)
    optimizer_log.addHandler(_optimizer_log_handler)
optimizer_log.setLevel(logging.INFO)

cpu_limiter = get_cpu_limiter()

FORWARD_TESTER_DURATION_MINUTES = 30
FORWARD_TESTER_PAUSE_SECONDS = 15
ML_TRAINER_CHECK_INTERVAL_SECONDS = 120
DATA_EXPORT_INTERVAL_SECONDS = 30
SQLITE_SAVE_INTERVAL_SECONDS = 45
OPTIMIZER_CYCLE_SECONDS = 14400.0
RESOURCE_MONITOR_INTERVAL_SECONDS = 5
PRUNER_INTERVAL_SECONDS = 86400  # once per day


# =============================================================================
# Forward-data loading (was run_forward_omlx_tester.py's load_klines_data)
# =============================================================================

async def load_klines_data(source: str = "paper", pair: str = "ETHUSDT", limit: int = 1000) -> list:
    """🔴 CRITICAL: LOAD REAL BYBIT DATA ONLY - NEVER SYNTHETIC, NEVER OTHER EXCHANGES

    This is a BYBIT bot. All data MUST come from BYBIT. No exceptions.
    If this function ever loads non-BYBIT data, the system will produce
    wrong results.
    """
    assert pair == "ETHUSDT", f"❌ FATAL: Only ETHUSDT allowed, got {pair}"
    log.info("Fetching %d REAL ETH/USDT candles from BYBIT...", limit)
    try:
        candles = await fetch_klines(pair, "1", limit)
        if candles and len(candles) > 0:
            log.info("✓ Loaded %d REAL ETH/USDT candles from BYBIT", len(candles))
            assert len(candles) > 0, "❌ FATAL: No candles loaded from BYBIT"
            assert len(candles[0]) >= 6, "❌ FATAL: Invalid candle format"
            return candles
    except Exception as e:
        log.error("❌ FATAL: Failed to fetch real BYBIT data: %s", e)
        raise RuntimeError(f"Cannot continue without BYBIT data: {e}")
    raise RuntimeError("❌ FATAL: Could not fetch real BYBIT data. System requires BYBIT for accuracy.")


# =============================================================================
# Continuous optimizer (was scripts/continuous_optimizer.py) — walk-forward
# parameter search. See that file's former module docstring, preserved
# here, for the full non-negotiable safety-gate rationale.
#
# Role in the bot: this decides HOW the DCA ladder is shaped (step %,
# take-profit, stop-loss/trailing, whether to stop-and-reverse) — a
# search over params, NOT over the OMLX decision layer itself (bounce
# scoring/ML stay fixed; only the mechanical ladder shape is searched).
# A promoted winner is written to the SQLite param library
# (optimization_store.py) and picked up from there by the PAPER TRADING
# BOT — symbot_python/api/paper.py polls it every PARAM_REFRESH_SECONDS
# (8h) and hot-applies whatever's currently promoted. This is the only
# path by which this loop's output reaches a running deal; it never
# touches a live deal directly.
# =============================================================================
#
# Every cycle:
#   1. Fetches your REAL available balance (read-only Keychain lookup —
#      never places an order) and a FRESH strict-14-day real market
#      window (never longer) for EACH candidate interval, plus real
#      funding-rate history and real risk-limit tiers.
#   2. Retests the CURRENT stored winner for each interval (if any)
#      against this fresh data — catches a winner that's stopped working
#      before searching for anything new. NON-NEGOTIABLE SAFETY GATE: if
#      the retest shows even one liquidation on this fresh data, the
#      winner is demoted immediately — a liquidation is never something
#      to wait out.
#   3. WALK-FORWARD validates a new candidate for each interval: rolls
#      through the same fetched 14-day window in WALK_FORWARD_IN_SAMPLE_DAYS
#      -in / WALK_FORWARD_OUT_SAMPLE_DAYS-out slices, re-optimizing every
#      roll by randomly sampling SEARCH_SAMPLES combinations from a large
#      grid — never on the same data it's about to be judged on.
#      NON-NEGOTIABLE SAFETY GATE, applied twice: walk_forward itself
#      never lets an in-sample-liquidating candidate through to its own
#      out-of-sample test, and the combined, genuinely out-of-sample
#      result is rejected outright if it liquidated even once.
#   4. Records every cycle's result to the SQLite param library.
#   5. Writes live progress to data/optimizer_status.json throughout.
#   6. Starts the next cycle OPTIMIZER_CYCLE_SECONDS after this cycle
#      started (or immediately if this cycle ran longer), fresh data
#      every time.

SYMBOL = "ETHUSDT"
INTERVALS = ["15", "30", "60", "240"]

SEARCH_GRID = ParamGrid(values={
    # No value below MIN_WIN_PRICE_PCT (0.33%): a take-profit target
    # smaller than the tiny-win floor can NEVER register a "real win" —
    # its own exit price is, by construction, under the raw-move
    # threshold is_real_win() requires.
    "dca_order_step_percent": [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.3, 2.0],
    "dca_order_size_multiplier": [1.0, 1.05, 1.1, 1.2, 1.3, 1.4],
    "leverage": [MIN_LEVERAGE, (MIN_LEVERAGE + MAX_LEVERAGE) / 2, MAX_LEVERAGE],
    "dca_max_order": [4, 5, 6, 8, 10, 12],
    "dca_order_step_percent_multiplier": [0.9, 1.0, 1.1],
    "dca_stop_loss_enabled": [False, True],
    "dca_stop_loss_percent": [1.5, 2.5, 4.0, 6.0, 10.0],
    "dca_trailing_stop_enabled": [False, True],
    "dca_trailing_stop_distance": [0.3, 0.5, 0.8, 1.2],
    "dca_trailing_activate_profit": [0.5, 1.0, 1.5, 2.0],
    "reverse_drawdown_percent": [None, 1.5, 2.5, 4.0, 6.0],
    "reverse_cooldown_sec": [1800.0, 3600.0, 7200.0],
    "max_consecutive_reversals": [1, 2, 3],
    "side": ["long", "short"],
    # Bounded by [MIN_FUNDS_UTILIZATION_PERCENT, MAX_FUNDS_UTILIZATION_PERCENT]
    # (funds_utilization_policy.py) — same single-source-of-truth pattern
    # as leverage above. A deal sized below the floor commits too little
    # of the account for even a fully-successful ladder to produce a
    # meaningful profit.
    "funds_utilization_percent": [
        MIN_FUNDS_UTILIZATION_PERCENT, 45.0, 60.0, 75.0, 90.0, MAX_FUNDS_UTILIZATION_PERCENT,
    ],
})

# The full grid has 447,897,600 combinations (SEARCH_GRID.total_combinations()
# — verified directly, the old comment here said 3,135,283,200, which no
# longer matched this grid) — exhaustively searching all of them EVERY
# WALK-FORWARD WINDOW would be far too slow. Random sampling a fixed
# number per window instead keeps wall-clock cost bounded.
SEARCH_SAMPLES = 100_000

# Every backtest/search/walk-forward run is capped at the real, strict
# DEFAULT_BACKTEST_DAYS (14) window. Splitting that fixed window into
# IN_SAMPLE_DAYS-in / OUT_SAMPLE_DAYS-out rolls (8/2, giving 3 rolls)
# means walk-forward validation happens entirely WITHIN the 14-day cap,
# never beyond it.
WALK_FORWARD_IN_SAMPLE_DAYS = 8
WALK_FORWARD_OUT_SAMPLE_DAYS = 2

STATUS_PATH = REPO_ROOT / "data" / "optimizer_status.json"


def write_optimizer_status(**fields) -> None:
    """Best-effort live-progress file for the web GUI to poll. Never
    raises — a status-reporting failure must never take the optimizer down.
    """
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if STATUS_PATH.exists():
            try:
                existing = json.loads(STATUS_PATH.read_text())
            except Exception:
                existing = {}
        existing.update(fields)
        existing["updated_at"] = time.time()
        tmp_path = STATUS_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(existing, default=str))
        os.replace(tmp_path, STATUS_PATH)
    except Exception:
        log.exception("Failed to write optimizer status file (non-fatal).")


async def fetch_fresh_market_data(session: HTTP, symbol: str, interval: str) -> tuple[list, list, list]:
    """Returns (candles, risk_tiers, funding_events) for exactly the
    locked-in DEFAULT_BACKTEST_DAYS window ending now."""
    limit = bars_for_days(interval, DEFAULT_BACKTEST_DAYS)
    candles = await fetch_klines(symbol, interval, limit)
    start_ts, end_ts = candles[0][0], candles[-1][0]

    risk_result = await call_with_timeout(
        asyncio.to_thread(session.get_risk_limit, category="linear", symbol=symbol)
    )
    risk_tiers = [
        (float(t["riskLimitValue"]), float(t["maxLeverage"]), float(t["maintenanceMargin"]))
        for t in risk_result["result"]["list"]
    ]

    funding_events: list[tuple[float, float]] = []
    cursor_end = end_ts
    while True:
        resp = await call_with_timeout(
            asyncio.to_thread(
                session.get_funding_rate_history, category="linear", symbol=symbol,
                endTime=int(cursor_end), limit=200,
            )
        )
        rows = resp["result"]["list"]
        if not rows:
            break
        for r in rows:
            ts = float(r["fundingRateTimestamp"])
            if ts >= start_ts:
                funding_events.append((ts, float(r["fundingRate"])))
        oldest = float(rows[-1]["fundingRateTimestamp"])
        if oldest <= start_ts or len(rows) < 200:
            break
        cursor_end = oldest - 1
    funding_events.sort()

    return candles, risk_tiers, funding_events


def make_base_config(price_tick: float, min_move: float, risk_tiers: list, funding_events: list) -> BacktestConfig:
    return BacktestConfig(
        first_order_amount=20.0, dca_order_amount=45.0, dca_max_order=8,
        dca_order_size_multiplier=1.08, dca_order_start_distance=0.5,
        dca_order_step_percent=0.5, dca_order_step_percent_multiplier=1.0,
        dca_take_profit_percent=FIXED_TAKE_PROFIT_PERCENT, exchange_fee=0.06,
        price_tick=price_tick, min_move_amount=min_move,
        leverage=MAX_LEVERAGE, maintenance_margin_rate=0.005, risk_tiers=risk_tiers,
        auto_size_to_funds=True, funds_utilization_percent=98.0,
        funding_events=funding_events,
    )


def optimizer_report_result(report) -> dict:
    return {
        "trades": len(report.trades),
        "win_rate": report.win_rate,
        "return_quote": report.total_profit_quote,
        "return_percent": report.total_profit_quote / report.starting_equity * 100 if report.starting_equity else 0.0,
        "max_dd_percent": report.max_drawdown_percent,
        "liquidations": report.liquidation_count,
        "sharpe": report.sharpe_ratio,
    }


def optimizer_record_kwargs(report, starting_equity: float) -> dict:
    """Full stat set for OptimizationRecord — same fields the manual
    backtest page shows, so the winners page can surface exactly as much
    detail as a fresh backtest run."""
    return dict(
        starting_equity=starting_equity,
        final_equity=report.final_equity,
        sharpe_ratio=report.sharpe_ratio,
        cagr_percent=report.cagr_percent,
        profit_factor=(report.profit_factor if report.profit_factor != float("inf") else None),
        gross_profit_quote=report.gross_profit_quote,
        gross_loss_quote=report.gross_loss_quote,
        max_win_quote=report.max_win_quote,
        max_loss_quote=report.max_loss_quote,
        max_win_percent=report.max_win_percent,
        max_loss_percent=report.max_loss_percent,
        average_win_quote=report.average_win_quote,
        average_loss_quote=report.average_loss_quote,
        average_win_percent=report.average_win_percent,
        average_loss_percent=report.average_loss_percent,
        max_win_streak=report.max_win_streak,
        max_loss_streak=report.max_loss_streak,
        total_fees_quote=report.total_fees_quote,
        total_funding_quote=report.total_funding_quote,
        liquidation_count=report.liquidation_count,
        average_trade_duration_hours=report.average_trade_duration_hours,
        average_safety_orders_used=report.average_safety_orders_used,
    )


async def run_optimizer_interval(
    conn, session: HTTP, symbol: str, interval: str, starting_equity: float, cycle_number: int,
) -> tuple[str, float, Optional[dict]]:
    """Runs retest + walk-forward validation for ONE interval. Returns
    (interval, best_score, summary)."""
    if not cpu_limiter.wait_until_safe(target_cpu=70.0, timeout_sec=120.0):
        optimizer_log.warning("[%s] CPU limiter timeout — deferring backtest to next cycle", interval)
        return interval, float("-inf"), None

    write_optimizer_status(state="fetching", symbol=symbol, interval=interval, cycle=cycle_number)
    candles, risk_tiers, funding_events = await fetch_fresh_market_data(session, symbol, interval)
    span_days = (candles[-1][0] - candles[0][0]) / 86_400_000
    optimizer_log.info("[%s] Fetched %d candles (%.2f days), %d funding events, %d risk tiers",
              interval, len(candles), span_days, len(funding_events), len(risk_tiers))

    info = await call_with_timeout(
        asyncio.to_thread(session.get_instruments_info, category="linear", symbol=symbol)
    )
    instrument = info["result"]["list"][0]
    price_tick = float(instrument["priceFilter"]["tickSize"])
    min_move = float(instrument["lotSizeFilter"]["qtyStep"])
    base_config = make_base_config(price_tick, min_move, risk_tiers, funding_events)

    # 1. Retest the current winner for this interval, if any — walk-
    # forward-FIXED, replaying its exact stored params across the SAME
    # rolling out-of-sample windows the challenger below gets evaluated
    # on, not a single full-window backtest (which would score it
    # against a fundamentally easier bar than the challenger's honest
    # walk-forward out-of-sample score).
    in_sample_bars = bars_for_days(interval, WALK_FORWARD_IN_SAMPLE_DAYS)
    out_sample_bars = bars_for_days(interval, WALK_FORWARD_OUT_SAMPLE_DAYS)

    current = get_current_winner(conn, symbol, interval)
    retest_summary = None
    if current is not None:
        winner_params = row_params(current)
        winner_config = BacktestConfig(**{**base_config.__dict__, **winner_params})
        retest_wf = walk_forward_fixed(
            winner_config, candles, starting_equity,
            in_sample_bars=in_sample_bars, out_sample_bars=out_sample_bars,
        )
        retest_report = retest_wf.combined_report()
        retest_score = score_lowest_loss_highest_pnl(retest_report)
        retest_summary = optimizer_report_result(retest_report)
        optimizer_log.info("[%s] Retest of current winner (walk-forward, %d window(s)): %s",
                  interval, len(retest_wf.windows), retest_summary)
        record_optimization(conn, OptimizationRecord(
            symbol=symbol, interval=interval, period_days=DEFAULT_BACKTEST_DAYS,
            params=winner_params, in_sample_score=retest_score,
            out_of_sample_return_quote=retest_report.total_profit_quote,
            out_of_sample_return_percent=retest_report.total_profit_quote / starting_equity * 100 if starting_equity else 0.0,
            out_of_sample_trade_count=len(retest_report.trades),
            out_of_sample_win_rate=retest_report.win_rate,
            max_drawdown_percent=retest_report.max_drawdown_percent,
            max_funds_required=0.0, run_kind="retest",
            **optimizer_record_kwargs(retest_report, starting_equity),
        ))
        # Non-negotiable safety gate: a winner that liquidates on FRESH
        # data is demoted immediately, not left running until something
        # better happens to come along.
        if retest_report.liquidation_count > 0:
            demote_winner(conn, symbol, interval)
            log.warning(
                "[%s] Current winner LIQUIDATED on retest (%d liquidation(s), fresh data) — "
                "demoted immediately, not left running.", interval, retest_report.liquidation_count,
            )
            retest_score = float("-inf")
    else:
        retest_score = float("-inf")
        optimizer_log.info("[%s] No current winner recorded yet.", interval)

    # 2. Walk-forward validate the search grid: roll through the fetched
    # 14-day window in (in-sample optimize) / (out-of-sample validate)
    # slices, re-optimizing every roll — the standard defense against
    # curve-fitting a strategy to the one window it happened to be
    # chosen on.
    estimated_windows = max(0, (len(candles) - in_sample_bars) // out_sample_bars)
    combos_total = estimated_windows * SEARCH_SAMPLES
    write_optimizer_status(state="searching", symbol=symbol, interval=interval, cycle=cycle_number,
                            combos_done=0, combos_total=combos_total, window_index=0, windows_total=estimated_windows)
    search_start = time.monotonic()
    windows_done = 0

    def on_combo_progress(done, total, best_so_far):
        write_optimizer_status(
            state="searching", symbol=symbol, interval=interval, cycle=cycle_number,
            combos_done=windows_done * SEARCH_SAMPLES + done, combos_total=combos_total,
            window_index=windows_done + 1, windows_total=estimated_windows,
            elapsed_seconds=time.monotonic() - search_start,
            best_so_far=(optimizer_report_result(best_so_far.report) if best_so_far else None),
        )

    def on_window(window_index, window):
        nonlocal windows_done
        windows_done = window_index + 1
        log.info(
            "[%s] walk-forward window %d: in-sample score %.2f, out-of-sample %s",
            interval, window_index + 1, window.in_sample_score, optimizer_report_result(window.out_sample_report),
        )

    wf_result = walk_forward(
        base_config, SEARCH_GRID, candles, starting_equity,
        in_sample_bars=in_sample_bars, out_sample_bars=out_sample_bars,
        scoring_fn=score_lowest_loss_highest_pnl,
        n_samples=SEARCH_SAMPLES, window_callback=on_window,
        combo_progress_callback=on_combo_progress, combo_progress_every=2000,
    )
    optimizer_log.info("[%s] Walk-forward finished in %.1fs (%d window(s))",
              interval, time.monotonic() - search_start, len(wf_result.windows))

    if not wf_result.windows or len(wf_result.windows) != estimated_windows:
        log.warning(
            "[%s] REJECTED: walk-forward did not complete every required window — "
            "nothing promoted, current winner (if any) unchanged.", interval,
        )
        return interval, retest_score, retest_summary

    combined_report = wf_result.combined_report()
    combined_score = score_lowest_loss_highest_pnl(combined_report)
    final_config = wf_result.windows[-1].best_config
    optimizer_log.info("[%s] Walk-forward combined out-of-sample result: %s", interval, optimizer_report_result(combined_report))

    # Non-negotiable promotion gate: NEVER promote a result that
    # liquidated even once across its OUT-OF-SAMPLE windows.
    if combined_report.liquidation_count > 0:
        log.warning(
            "[%s] REJECTED: walk-forward result liquidated %d time(s) out-of-sample — "
            "nothing promoted, current winner (if any) unchanged.",
            interval, combined_report.liquidation_count,
        )
        return interval, retest_score, retest_summary

    result_id = record_optimization(conn, OptimizationRecord(
        symbol=symbol, interval=interval, period_days=DEFAULT_BACKTEST_DAYS,
        params={k: getattr(final_config, k) for k in SEARCH_GRID.values},
        in_sample_score=wf_result.windows[-1].in_sample_score,
        out_of_sample_return_quote=combined_report.total_profit_quote,
        out_of_sample_return_percent=combined_report.total_profit_quote / starting_equity * 100 if starting_equity else 0.0,
        out_of_sample_trade_count=len(combined_report.trades),
        out_of_sample_win_rate=combined_report.win_rate,
        max_drawdown_percent=combined_report.max_drawdown_percent,
        max_funds_required=0.0, run_kind="walk_forward",
        **optimizer_record_kwargs(combined_report, starting_equity),
    ))

    if combined_score > retest_score:
        promote_to_winner(conn, symbol, interval, result_id)
        log.info(
            "[%s] New winner promoted (walk-forward score %.2f > previous %.2f), "
            "zero out-of-sample liquidations confirmed across %d window(s).",
            interval, combined_score, retest_score, len(wf_result.windows),
        )
        return interval, combined_score, optimizer_report_result(combined_report)
    else:
        optimizer_log.info("[%s] Current winner still holds (score %.2f >= new walk-forward result %.2f).",
                  interval, retest_score, combined_score)
        return interval, retest_score, retest_summary


async def run_optimizer_cycle(session: HTTP, cycle_number: int) -> None:
    cycle_start = time.monotonic()
    log.info("=== Cycle %d start: %s, real balance, intervals=%s ===", cycle_number, SYMBOL, INTERVALS)

    # Always the real, current available balance — never a placeholder,
    # UNLESS it's effectively dust (see paper_module.MIN_USABLE_BALANCE_USDT's
    # own comment: a real account can carry sub-cent leftovers like
    # $0.0000390 from fees/funding that are > 0 but useless as a backtest
    # sizing base). Fresh every cycle: this is a shared account (the live
    # bot also trades on it), so "available" moves as that bot's own
    # positions open/close. Read-only — never places an order, only sizes
    # the backtest realistically.
    #
    # Real bug this fixes: with no floor, starting_equity could be
    # ~0.00003904 — every walk-forward window's return_percent
    # (profit_quote / starting_equity * 100) then explodes into the tens
    # of millions of percent (observed directly: -88,759,647%), which is
    # exactly why the optimizer looked like it was "showing nothing" on
    # the dashboard — paper.py's own get_manager() docstring already
    # claimed this fallback was shared with "the continuous optimizer";
    # it never actually was until now.
    balance = await fetch_real_balance()
    starting_equity = balance.total_available_balance
    if starting_equity < paper_module.MIN_USABLE_BALANCE_USDT:
        log.warning(
            "Real available balance is %.8f USDT — walk-forward sizing would be "
            "meaningless. Falling back to a simulated %.2f USDT starting equity "
            "instead (no real funds involved, same fallback paper trading uses).",
            starting_equity, paper_module.FALLBACK_STARTING_BALANCE_USDT,
        )
        starting_equity = paper_module.FALLBACK_STARTING_BALANCE_USDT
    log.info("Real account balance: totalEquity=%.2f totalAvailableBalance=%.2f (starting_equity used=%.2f)",
              balance.total_equity, balance.total_available_balance, starting_equity)
    write_optimizer_status(state="starting", cycle=cycle_number, symbol=SYMBOL,
                            real_total_equity=balance.total_equity,
                            real_available_balance=balance.total_available_balance,
                            intervals=INTERVALS)

    conn = connect_optimizer_store()
    per_interval_best: dict[str, tuple[float, Optional[dict]]] = {}
    try:
        for interval in INTERVALS:
            # Isolate one interval's failure (a transient API error, a
            # malformed response, a timeout) from the other three — this
            # runs unattended for weeks, and one flaky network call must
            # never cost 3 healthy intervals their update this cycle
            # (every OPTIMIZER_CYCLE_SECONDS, 4 hours by default).
            try:
                _, score, summary = await run_optimizer_interval(
                    conn, session, SYMBOL, interval, starting_equity, cycle_number
                )
                per_interval_best[interval] = (score, summary)
            except Exception:
                optimizer_log.exception("[%s] Interval failed this cycle — skipping, other intervals unaffected.", interval)
                write_optimizer_status(state="error", symbol=SYMBOL, interval=interval, cycle=cycle_number)
    finally:
        conn.close()

    if per_interval_best:
        winning_interval, (winning_score, winning_summary) = max(
            per_interval_best.items(), key=lambda kv: kv[1][0]
        )
        log.info(">>> Best interval this cycle: %s (score=%.2f) %s", winning_interval, winning_score, winning_summary)
        write_optimizer_status(
            state="sleeping", cycle=cycle_number,
            last_cycle_winning_interval=winning_interval,
            last_cycle_winning_summary=winning_summary,
            last_cycle_seconds=time.monotonic() - cycle_start,
        )
    else:
        write_optimizer_status(state="sleeping", cycle=cycle_number, last_cycle_seconds=time.monotonic() - cycle_start)

    log.info("Cycle %d complete in %.1fs total.\n", cycle_number, time.monotonic() - cycle_start)


def _run_optimizer_cycle_sync(session: HTTP, cycle_number: int) -> None:
    """Runs one optimizer cycle on a private event loop, on a worker
    thread. run_optimizer_interval()'s walk_forward() call is a plain
    synchronous function (no internal awaits during the actual backtest
    loop, up to 100,000 sampled combos per walk-forward window) — that
    CPU-bound work would otherwise run directly on this process's shared
    event loop and starve everything else sharing it (uvicorn included).
    asyncio.run() on a worker thread gives this cycle its own event loop
    entirely, isolated from the one driving uvicorn/the other loops.
    """
    asyncio.run(run_optimizer_cycle(session, cycle_number))


async def optimizer_loop(cycle_seconds: float = OPTIMIZER_CYCLE_SECONDS) -> None:
    session = HTTP()  # unauthenticated: public market/risk/funding data only, never trades
    cycle_number = 0
    # The first cycle starts immediately, so a request left over from a
    # previous run is already satisfied by that cycle.
    consume_optimizer_run_request()
    while True:
        cycle_number += 1
        started = time.monotonic()
        try:
            await asyncio.to_thread(_run_optimizer_cycle_sync, session, cycle_number)
        except asyncio.CancelledError:
            raise
        except Exception:
            optimizer_log.exception("[optimizer] cycle %d failed — retrying next cycle", cycle_number)
        remaining = max(0.0, cycle_seconds - (time.monotonic() - started))
        write_optimizer_status(next_cycle_in_seconds=int(remaining))
        optimizer_log.info("[optimizer] sleeping %.0fs until next cycle (or a bot-restart request)", remaining)
        for _ in range(int(remaining)):
            if consume_optimizer_run_request():
                optimizer_log.info("[optimizer] bot-restart requested a fresh cycle")
                break
            await asyncio.sleep(1)


# =============================================================================
# Forward tester (was run_forward_omlx_tester.py)
#
# Role in the bot: this is the ONLY source of training data for the ML
# layer (ml_trainer_loop below) and the only way OMLX's bounce-scoring
# gets tested against real, current market conditions before real money
# ever sees it. Each cycle replays real ETHUSDT candles through the full
# decision stack (dip detection -> 10-dimension bounce score -> ML-
# blended action) across many parameter combinations, and every trade it
# generates gets written to forward_test_accuracy_*.json — which
# ml_trainer_loop below reads and folds into trade_memory.json. No
# synthetic data path exists anywhere in this loop (see the FATAL
# asserts in load_klines_data above) — if this loop's real-BYBIT-data
# guarantee is ever weakened, every model downstream starts learning
# from fake data without any other signal that something changed.
# =============================================================================

async def _forward_tester_cycle_async(duration_minutes: int) -> dict:
    klines = await load_klines_data("paper", "ETHUSDT", 1000)
    return await run_forward_tester(klines, duration_minutes=duration_minutes)


def _run_forward_tester_cycle_sync(duration_minutes: int) -> dict:
    """Same reasoning as _run_optimizer_cycle_sync: run_forward_tester()
    tests many parameter combinations synchronously (CPU-bound), which
    would otherwise starve the shared event loop — given its own private
    loop on a worker thread instead."""
    return asyncio.run(_forward_tester_cycle_async(duration_minutes))


async def forward_tester_loop(
    duration_minutes: int = FORWARD_TESTER_DURATION_MINUTES,
    pause_seconds: int = FORWARD_TESTER_PAUSE_SECONDS,
) -> None:
    while True:
        try:
            report = await asyncio.to_thread(_run_forward_tester_cycle_sync, duration_minutes)
            output_file = REPO_ROOT / f"forward_test_report_{int(datetime.now().timestamp())}.json"
            output_file.write_text(json.dumps(report, indent=2, default=str))
            log.info("[forward_tester] cycle complete → %s", output_file.name)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[forward_tester] cycle failed — retrying after pause")
        await asyncio.sleep(pause_seconds)


# =============================================================================
# ML trainer (symbot_python/ml/continuous_trainer.py — already a proper
# library class; this just runs its blocking loop off the event loop)
#
# Role in the bot: this is the "learn" half of the loop forward_tester_loop
# feeds. Every check_interval, ContinuousMLTrainer folds any NEW
# forward-test reports into trade_memory.json (deduped — see TradeMemory.
# processed_reports, and the MAX_TRADES=20_000 cap; both exist because
# this exact loop, unbounded, once grew trade_memory.json to 5.7GB — see
# CLAUDE.md's "Recent Significant Changes"), then retrains
# outcome_predictor.py (XGBoost: given these features, will this trade
# win?) and rl_decision_optimizer.py (Q-learning: in this market state,
# which action — add safety / wait / bail — actually paid off?) on the
# updated trade history. omlx_ml_advisor.py (called from inside the
# forward tester's/live engine's decision path, NOT from this file)
# loads whatever these two models currently say and blends it with the
# bounce analyzer's own score. This is the entire "gets smarter over
# time" property of the bot — nothing else in this file trains anything.
# =============================================================================

async def ml_trainer_loop(check_interval: int = ML_TRAINER_CHECK_INTERVAL_SECONDS) -> None:
    trainer = ContinuousMLTrainer(check_interval_seconds=check_interval)
    await asyncio.to_thread(trainer.run_forever)


# =============================================================================
# Data exporter (was continuous_data_exporter.py + export_trading_data.py's
# TradingDataExporter class)
#
# Role in the bot: none directly — this is read-only reporting. Reads
# whatever paper_trade_*.json / backtest_*.json / positions.json already
# exist and reshapes them into the *_summary.json files the dashboard's
# JS actually fetches. If a dashboard number looks wrong, check whether
# the SOURCE file this reads from is stale/wrong first — this never
# invents data, only reformats what's already on disk.
# =============================================================================

def _atomic_write_json(filename: str, data: dict) -> None:
    """Temp file + os.replace, same reasoning as trade_memory.py's
    save()/dip_calibration_engine.py's _save_state()/this file's own
    _export_omlx_metrics_once(). Real bug this fixes: these 4 dashboard
    JSON files are now actually served (app.py's explicit routes,
    FileResponse) as of this same fix — previously they were written
    with a plain `open(..., "w")` and never read by anything, so a
    concurrent read racing a write was impossible. FileResponse computes
    Content-Length from the file's size at request start, then streams
    it; export_all() rewriting the file mid-stream (every 30s) could
    make the actual bytes served exceed that Content-Length, which
    surfaced immediately after this fix landed as a real
    "RuntimeError: Response content longer than Content-Length" in
    production. os.replace() is atomic on POSIX, so a concurrent
    FileResponse only ever sees the old complete file or the new
    complete file, never a partial write.
    """
    tmp_path = f"{filename}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filename)


class TradingDataExporter:
    """Export trading metrics to JSON — real data pulled directly from
    the in-process paper trading manager (symbot_python/api/paper.py),
    not file-mediated hand-off from a separate process (there is none;
    everything is this one process).

    Real bug this class's original three methods below all shared:
    export_paper_trading_data()/export_positions()/export_reverse_trading()
    each looked for files (paper_trade_*.json / positions.json /
    reverse_trades.json) that NOTHING in this codebase has ever written
    — not before the single-script merge, not after. Confirmed directly
    (grep across the whole repo): zero write sites for any of the three.
    export_paper_trading_data() and the old export_reverse_trading()
    silently fell through to an honest all-zeros default every single
    call, forever — that part was at least not misleading, just always
    empty. export_positions() was worse: it wrote a HARDCODED FAKE
    position ("ETH, size 1.5, entry $3250.50...") to positions.json the
    first time it ran, then kept re-reading and re-serving that same
    fake file forever afterward, since Path("positions.json").exists()
    was true from then on — a real user could watch the Position
    Tracking tab and see what looked like a genuine open position that
    never existed. All three now pull real state from
    trading_module.get_manager() (and _reverse_paper) — an empty list/
    zero counts when nothing is genuinely open is the honest result,
    not something to paper over with placeholder data.
    """

    def __init__(self):
        self.data_dir = Path(".")

    async def export_paper_trading_data(self):
        """Export real paper trading session results."""
        try:
            manager = await trading_module.get_manager()
            closed = [
                d for d in manager.deals.values()
                if d.status == DealStatus.CLOSED and d.sell_data
            ]
            total = len(closed)
            if total == 0:
                self._write_default_paper()
                return

            pnls = [d.sell_data["profit_quote"] for d in closed]
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / total * 100
            total_pnl = sum(pnls)
            running, peak, max_dd = 0.0, 0.0, 0.0
            for d in sorted(closed, key=lambda d: d.date_opened):
                running += d.sell_data["profit_quote"]
                peak = max(peak, running)
                if peak > 0:
                    max_dd = max(max_dd, (peak - running) / peak * 100)

            recent = sorted(closed, key=lambda d: d.date_opened, reverse=True)[:5]
            data = {
                "total_trades": total,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "max_drawdown": max_dd,
                "recent_trades": [
                    {
                        "entry": f"${(d.orders[0].average if d.orders else 0):.2f}",
                        "exit": f"${d.sell_data.get('price', 0):.2f}",
                        "pnl": d.sell_data["profit_quote"],
                        # Raw epoch seconds — dashboard JS renders it with
                        # Date().toLocaleString(), i.e. the viewer's own
                        # system time/timezone, same convention already
                        # used for the OMLX Learning tab's "Last
                        # Calibrated"/"Last Trained" timestamps. Real gap
                        # this fixes: recent_trades never carried a
                        # timestamp field at all — the dashboard's Recent
                        # Trades table has always had no way to show WHEN
                        # a trade closed, only its prices/PnL.
                        "closed_at": d.date_closed,
                    }
                    for d in recent
                ],
            }
            _atomic_write_json("paper_trading_summary.json", data)
            log.info("✓ Paper trading: %d trades, %.1f%% win rate", total, win_rate)
        except Exception as e:
            log.error("Error exporting paper data: %s", e)
            self._write_default_paper()

    def export_backtest_data(self):
        """Export the real continuous optimizer's best current winner.

        Previously read backtest_*.json — files only ever produced by a
        manual single-backtest web form deleted in an earlier cleanup
        pass (see CLAUDE.md's "Recent Significant Changes", 2026-09-17:
        "removed manual-backtest web UI"). Those files never existed
        under the current single-script architecture, so this always
        silently fell through to _write_default_backtest() (all zeros)
        — not just on the dashboard (nothing read backtest_summary.json
        there either, until the "Backtest Results" tab was rewired to
        /api/optimizer-status), but into ethladder_analytics.db's
        backtest_results table too, every 45 seconds, forever, via
        sqlite_saver_loop → _sqlite_save_once → save_backtest_results.
        """
        try:
            conn = connect_optimizer_store()
            try:
                row = get_best_current_winner(conn, SYMBOL)
            finally:
                conn.close()

            if row is None:
                self._write_default_backtest()
                return

            winner = dict(row)
            data = {
                "total_trades": winner.get("out_of_sample_trade_count", 0) or 0,
                "total_return": winner.get("out_of_sample_return_percent", 0) or 0,
                "win_rate": (winner.get("out_of_sample_win_rate", 0) or 0) * 100,
                "sharpe_ratio": winner.get("sharpe_ratio", 0) or 0,
                "profit_factor": winner.get("profit_factor") if winner.get("profit_factor") is not None else 0,
                "max_drawdown": winner.get("max_drawdown_percent", 0) or 0,
                "avg_win": winner.get("average_win_percent", 0) or 0,
                "avg_loss": winner.get("average_loss_percent", 0) or 0,
                "recovery_factor": (
                    winner.get("out_of_sample_return_percent", 0) / winner.get("max_drawdown_percent")
                    if winner.get("max_drawdown_percent") else 0
                ),
            }
            _atomic_write_json("backtest_summary.json", data)
            log.info("✓ Backtest: %d trades, %.2f%% return (interval %s)",
                      data["total_trades"], data["total_return"], winner.get("interval"))
        except Exception as e:
            log.error("Error exporting backtest: %s", e)
            self._write_default_backtest()

    async def export_positions(self):
        """Export currently-open paper deals AND live Bybit positions."""
        try:
            positions = []
            current_price = None

            # Get ticker once for unrealized PnL calculations
            try:
                paper_mgr = await trading_module.get_manager()
                ticker = await paper_mgr.exchange.get_ticker(SYMBOL)
                current_price = ticker.last
            except Exception:
                current_price = None

            # Export paper trading positions
            try:
                paper_mgr = await trading_module.get_manager()
                open_deals = [d for d in paper_mgr.deals.values() if d.status == DealStatus.ACTIVE]
                for deal in open_deals:
                    bot = deal.config or paper_mgr.bots.get(deal.bot_id)
                    if not deal.filled_count or bot is None:
                        continue
                    last_filled = deal.orders[deal.filled_count - 1]
                    unrealized_pnl = None
                    if current_price is not None:
                        sign = -1 if bot.side == "short" else 1
                        unrealized_pnl = sign * last_filled.qty_sum * (current_price - last_filled.average)
                    positions.append({
                        "deal_id": deal.deal_id,
                        "symbol": bot.pair,
                        "side": bot.side,
                        "size": last_filled.qty_sum,
                        "entry_price": last_filled.average,
                        "current_price": current_price,
                        "unrealized_pnl": unrealized_pnl,
                        "status": "open",
                        "trading_mode": "paper",
                    })
            except Exception as e:
                log.warning("Error exporting paper positions: %s", e)

            # Export live trading positions from Bybit (only if manager exists, don't auto-init)
            try:
                if trading_module._manager is not None:
                    live_mgr = trading_module._manager
                    live_open_deals = [d for d in live_mgr.deals.values() if d.status == DealStatus.ACTIVE]
                    for deal in live_open_deals:
                        bot = deal.config or live_mgr.bots.get(deal.bot_id)
                        if not deal.filled_count or bot is None:
                            continue
                        last_filled = deal.orders[deal.filled_count - 1]
                        unrealized_pnl = None
                        if current_price is not None:
                            sign = -1 if bot.side == "short" else 1
                            unrealized_pnl = sign * last_filled.qty_sum * (current_price - last_filled.average)
                        positions.append({
                            "deal_id": deal.deal_id,
                            "symbol": bot.pair,
                            "side": bot.side,
                            "size": last_filled.qty_sum,
                            "entry_price": last_filled.average,
                            "current_price": current_price,
                            "unrealized_pnl": unrealized_pnl,
                            "status": "open",
                            "trading_mode": "live",
                        })
            except Exception as e:
                log.warning("Error exporting live positions: %s", e)

            _atomic_write_json("positions.json", {"positions": positions})
            log.info("✓ Positions: %d paper, %d live",
                    sum(1 for p in positions if p.get("trading_mode") == "paper"),
                    sum(1 for p in positions if p.get("trading_mode") == "live"))
        except Exception as e:
            log.error("Error exporting positions: %s", e)

    async def export_reverse_trading(self):
        """Export real reverse-paper-trading state (the opposite-side
        mirror wallet — symbot_python/exchange/reverse_paper.py)."""
        try:
            reverse = trading_module._reverse_paper
            if reverse is None:
                data = {"total_trades": 0, "win_rate": None, "total_pnl": 0, "conversion_rate": None, "reversal_patterns": []}
            else:
                manager = await trading_module.get_manager()
                ticker = await manager.exchange.get_ticker(SYMBOL)
                position = reverse.client.position(SYMBOL)
                unrealized = position.qty * (ticker.last - position.avg_price) if position.qty else 0.0
                equity = reverse.client.margin_balance + position.margin_committed + unrealized
                fills = reverse.fills
                failed = sum(1 for fill in fills if fill.error is not None)
                data = {
                    "total_trades": len(fills),
                    "win_rate": None,  # fills are individual order mirrors, not paired closed trades — no win/loss label to give honestly
                    "total_pnl": equity - reverse.initial_balance,
                    "conversion_rate": ((len(fills) - failed) / len(fills) * 100) if fills else None,
                    "failed_mirror_count": failed,
                    "current_balance": reverse.client.margin_balance,
                    "current_equity": equity,
                    "reversal_patterns": [],
                }
            _atomic_write_json("reverse_trading_summary.json", data)
            log.info("✓ Reverse trading: %d fills", data["total_trades"])
        except Exception as e:
            log.error("Error exporting reverse data: %s", e)

    def _write_default_paper(self):
        data = {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "max_drawdown": 0, "recent_trades": []}
        _atomic_write_json("paper_trading_summary.json", data)

    def _write_default_backtest(self):
        data = {
            "total_trades": 0, "total_return": 0, "win_rate": 0, "sharpe_ratio": 0,
            "profit_factor": 0, "max_drawdown": 0, "avg_win": 0, "avg_loss": 0, "recovery_factor": 0,
        }
        _atomic_write_json("backtest_summary.json", data)

    async def export_all(self):
        """Export all data. Async now — every method here just reads
        already-in-memory manager state plus at most one ticker fetch
        (already-async I/O) or a fast local SQLite read, none of it
        CPU-bound, so this runs directly on the shared event loop rather
        than via asyncio.to_thread() (unlike the optimizer/forward-tester,
        which genuinely are CPU-bound and must stay off it)."""
        log.info("Exporting trading data...")
        await self.export_paper_trading_data()
        self.export_backtest_data()
        await self.export_positions()
        log.info("✓ All trading data exported")


async def data_exporter_loop(interval_seconds: int = DATA_EXPORT_INTERVAL_SECONDS) -> None:
    exporter = TradingDataExporter()
    while True:
        try:
            if cpu_limiter.check_and_throttle("data_export"):
                await exporter.export_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[data_exporter] export failed")
        await asyncio.sleep(interval_seconds)


# =============================================================================
# OMLX metrics exporter (was export_live_metrics.py, supervised as
# "Metrics Exporter" by the old supervise_system.sh) — reads the latest
# forward-test accuracy/calibration reports and writes
# omlx_metrics_live.json, which the dashboard's OMLX Learning tab polls.
# =============================================================================

def _get_ml_trainer_status() -> dict:
    """Read ContinuousMLTrainer.get_status(), persisted by
    continuous_trainer.py's _save_status() every check-interval — this is
    the fix for the dashboard's "Training Cycles" tile, which previously
    read progress.training_cycles, a key _get_current_omlx_metrics()
    below never actually set (always displayed "0" regardless of real
    training activity)."""
    path = REPO_ROOT / "ml_trainer_status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _get_calibration_status() -> dict:
    """Read DipCalibrationEngine.get_status_summary() from its persisted
    state file directly (not via the async get_calibration_engine()
    singleton — this function runs synchronously off the event loop via
    asyncio.to_thread()). This is the real, continuously-updating
    decision→outcome→recalibrate loop (both live/paper trading and the
    walk-forward forward-tester now feed the SAME shared engine, tagged
    by DipTradeRecord.source) — previously invisible on the dashboard
    entirely; the only calibration data ever exported
    (forward_test_calibration_*.json) came from a throwaway instance in
    forward_omlx_tester.py that nothing ever called record_decision/
    record_outcome on, so it was always empty defaults.
    """
    path = REPO_ROOT / "dip_calibration_state.json"
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except Exception:
        return {}
    trades = state.get("trades", [])
    training_events = state.get("training_events", [])
    completed = [t for t in trades if t.get("outcome_bounced") is not None]
    by_source: dict[str, int] = {}
    for t in completed:
        by_source[t.get("source", "unknown")] = by_source.get(t.get("source", "unknown"), 0) + 1
    last_event = training_events[-1] if training_events else None
    return {
        "total_decisions": len(trades),
        "completed_decisions": len(completed),
        "completed_by_source": by_source,
        "overall_bounce_rate": (
            sum(1 for t in completed if t.get("outcome_bounced")) / len(completed)
            if completed else None
        ),
        "patterns_tracked": len(state.get("pattern_success_rates", {})),
        "recommended_weights": state.get("recommended_weights", {}),
        "training_event_count": len(training_events),
        "last_calibrated_at": last_event["timestamp"] if last_event else None,
        "recent_events": training_events[-20:],
        "recent_decisions": trades[-20:],
    }


def _get_current_omlx_metrics() -> dict:
    current: dict = {}
    current["ml_trainer"] = _get_ml_trainer_status()
    current["inbox_candidates"] = inbox_pattern_extractor.get_status()
    # This IS the real calibration data — dip_calibration_state.json is
    # updated on every record_outcome() call from either source (live/
    # paper trading or the walk-forward forward-tester), continuously,
    # not just once per 30-min forward-test cycle. A previous version of
    # this function tried reading forward_test_calibration_*.json's
    # top-level "weights" key here instead — that key never existed in
    # export_calibration_data()'s output (the real keys are
    # pattern_success_rates/dimension_accuracy/recommended_weights/etc.),
    # so it always silently returned {} regardless of real calibration
    # state, on top of that export coming from a throwaway
    # DipCalibrationEngine() instance forward_omlx_tester.py never
    # actually recorded anything into (see its own fix's commit message).
    current["calibration"] = _get_calibration_status()

    accuracy_files = sorted(Path(".").glob("forward_test_accuracy_*.json"), reverse=True)
    if accuracy_files:
        with open(accuracy_files[0]) as f:
            acc_data = json.load(f)
            metrics = acc_data.get("metrics", {})
            # total_trades lives at the TOP level as total_trades_analyzed
            # (see OMLXAccuracyOptimizer.export_accuracy_report) — it was
            # never a key inside metrics at all, so metrics.get("total_trades", 0)
            # silently read 0 for every export regardless of real trade
            # count. Likewise pattern data is metrics["patterns"] (built
            # by get_accuracy_report() from failed_patterns), not a
            # top-level "pattern_success_rates" key, which never existed.
            current["metrics"] = {
                "accuracy": metrics.get("accuracy", 0) * 100,
                "win_rate": metrics.get("win_rate", 0) * 100,
                "precision": metrics.get("precision", 0) * 100,
                "recall": metrics.get("recall", 0) * 100,
                "f1_score": metrics.get("f1_score", 0),
                "avg_win": metrics.get("avg_win", 0),
                "avg_loss": metrics.get("avg_loss", 0),
                "total_trades": acc_data.get("total_trades_analyzed", 0),
                "accuracy_change": metrics.get("accuracy_change", 0),
                "winrate_change": metrics.get("winrate_change", 0),
                "precision_change": metrics.get("precision_change", 0),
                "f1_change": metrics.get("f1_change", 0),
                "patterns_calibrated": len(metrics.get("patterns", {})),
            }
            current["patterns"] = metrics.get("patterns", {})
            current["rules"] = acc_data.get("loss_prevention_rules", [])

    # Real phase thresholds derived from the calibration engine's actual
    # completed-decision count — was a hardcoded array that always
    # reported "Loss Minimization... current" regardless of real
    # progress (even on a completely fresh, zero-trade install).
    completed = current["calibration"].get("completed_decisions", 0)
    current["phases"] = [
        {
            "name": "Initial Calibration",
            "description": "Baseline patterns learned",
            "status": "complete" if completed >= 30 else ("current" if completed > 0 else "pending"),
        },
        {
            "name": "Pattern Confidence",
            "description": "100+ completed decisions across both sources",
            "status": "complete" if completed >= 100 else ("current" if completed >= 30 else "pending"),
        },
        {
            "name": "Loss Minimization",
            "description": "500+ completed decisions, dimension weights stabilizing",
            "status": "complete" if completed >= 500 else ("current" if completed >= 100 else "pending"),
        },
        {
            "name": "Mature Calibration",
            "description": "2,000+ completed decisions",
            "status": "current" if completed >= 500 else "pending",
        },
    ]
    current_phase = next(
        (p for p in current["phases"] if p["status"] == "current"),
        current["phases"][0] if completed == 0 else current["phases"][-1],
    )

    accuracy = current.get("metrics", {}).get("accuracy", 0)
    ml_trainer = current["ml_trainer"]
    current["progress"] = {
        "overall_progress": min(accuracy, 100),
        "completed_decisions": completed,
        "patterns_calibrated": current["calibration"].get("patterns_tracked", 0),
        "rules_generated": len(current.get("rules", [])),
        "phase_label": current_phase["name"],
        # The actual fix for the dashboard's previously-dead "Training
        # Cycles" tile (metrics.progress?.training_cycles in the
        # dashboard JS) — this key was never set anywhere before.
        "training_cycles": ml_trainer.get("training_cycles", 0),
        "last_training_time": ml_trainer.get("last_training"),
    }
    return current


def _export_omlx_metrics_once() -> None:
    metrics_data = _get_current_omlx_metrics()
    if "metrics" not in metrics_data:
        metrics_data["metrics"] = {}
    metrics_data["last_updated"] = datetime.now().isoformat()
    tmp_path = REPO_ROOT / "omlx_metrics_live.json.tmp"
    tmp_path.write_text(json.dumps(metrics_data, indent=2))
    os.replace(tmp_path, REPO_ROOT / "omlx_metrics_live.json")


async def omlx_metrics_exporter_loop(interval_seconds: int = DATA_EXPORT_INTERVAL_SECONDS) -> None:
    while True:
        try:
            if cpu_limiter.check_and_throttle("omlx_metrics_export"):
                await asyncio.to_thread(_export_omlx_metrics_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[omlx_metrics_exporter] export failed")
        await asyncio.sleep(interval_seconds)


# =============================================================================
# SQLite saver (was continuous_sqlite_saver.py + data_persistence.py's
# DataPersistence class)
#
# Role in the bot: none directly — this is the ONLY thing that writes to
# ethladder_analytics.db (the long-term historical record the dashboard's
# Analytics tab and analyze_history.py query). It reads the same *.json
# snapshot files data_exporter/omlx_metrics_exporter/resource_monitor
# just wrote and appends a timestamped row per table — the JSON files
# are "current state", this DB is "history over time".
# =============================================================================

import sqlite3  # noqa: E402


class DataPersistence:
    """Save all metrics to SQLite."""

    def __init__(self, db_path: str = "ethladder_analytics.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        # No longer connects here — see connect()'s docstring for why.

    def connect(self):
        """Open a fresh connection. Real bug this fixes: this class used
        to connect ONCE in __init__ (on the event-loop thread) and hold
        that same connection for the rest of the process's life, reused
        every cycle via asyncio.to_thread() — which does NOT guarantee
        the same OS thread services every call (its default executor is
        a POOL, and this process has many concurrent asyncio.to_thread()
        callers competing for it: optimizer, forward_tester, ml_trainer,
        resource_monitor...). check_same_thread=False only disables
        Python's OWN guard against this; it does not make the underlying
        C sqlite3 library safe for a connection object to migrate
        between OS threads across calls on every build. Confirmed
        directly: a real SIGSEGV crashed the whole process, faulting
        thread "ThreadPoolExecutor-0_4", top frame
        _pysqlite_query_execute — exactly this connection, reused across
        the pool. Fixed by opening a fresh connection inside
        _sqlite_save_once() itself, entirely within ONE
        asyncio.to_thread() call — its whole lifecycle now stays on
        whichever single OS thread services that one call, never
        spanning two.
        """
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS omlx_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL, win_rate REAL, total_trades INTEGER,
                    patterns_count INTEGER, training_cycles INTEGER,
                    confidence REAL, rules_count INTEGER
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL, memory_used_gb REAL, memory_total_gb REAL,
                    memory_percent REAL, disk_percent REAL, disk_used_gb REAL,
                    disk_total_gb REAL, temperature_celsius REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    entry_price REAL, exit_price REAL, pnl REAL,
                    win INTEGER, trade_size REAL, duration_seconds INTEGER
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_trades INTEGER, win_rate REAL, total_return REAL,
                    sharpe_ratio REAL, profit_factor REAL, max_drawdown REAL,
                    recovery_factor REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT, entry_price REAL, current_price REAL,
                    size REAL, unrealized_pnl REAL, status TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pattern_name TEXT, success_rate REAL, occurrences INTEGER,
                    avg_pnl REAL, confidence REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE, avg_accuracy REAL, avg_win_rate REAL,
                    avg_cpu REAL, avg_memory REAL, avg_temperature REAL,
                    trades_executed INTEGER, total_pnl REAL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_omlx_timestamp ON omlx_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON paper_trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date)')
            self.conn.commit()
            log.info("✓ Database initialized")
        except Exception as e:
            log.error("Database init error: %s", e)

    def save_omlx_metrics(self, metrics: dict):
        try:
            if not self.conn:
                log.debug("Reconnecting to database for OMLX metrics")
                self.connect()
            if not self.conn:
                log.error("Database connection failed after reconnect attempt")
                return
            cursor = self.conn.cursor()
            m = metrics.get("metrics", {})
            p = metrics.get("progress", {})
            cursor.execute('''
                INSERT INTO omlx_metrics
                (accuracy, win_rate, total_trades, patterns_count, training_cycles, rules_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                m.get("accuracy", 0), m.get("win_rate", 0), m.get("total_trades", 0),
                len(metrics.get("patterns", {})), p.get("training_cycles", 0), len(metrics.get("rules", [])),
            ))
            self.conn.commit()
        except Exception as e:
            log.error("Error saving OMLX metrics: %s", e)

    def save_system_metrics(self, metrics: dict):
        try:
            if not self.conn:
                self.connect()
            cursor = self.conn.cursor()
            current = metrics.get("current", {})
            cpu = current.get("cpu", {})
            mem = current.get("memory", {})
            disk = current.get("disk", {})
            temp = current.get("temperature", {})
            cursor.execute('''
                INSERT INTO system_metrics
                (cpu_percent, memory_used_gb, memory_total_gb, memory_percent,
                 disk_percent, disk_used_gb, disk_total_gb, temperature_celsius)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cpu.get("percent", 0), mem.get("used_gb", 0), mem.get("total_gb", 0), mem.get("percent", 0),
                disk.get("percent", 0), disk.get("used_gb", 0), disk.get("total_gb", 0), temp.get("celsius", None),
            ))
            self.conn.commit()
        except Exception as e:
            log.error("Error saving system metrics: %s", e)

    def save_paper_trades(self, trades: list):
        try:
            if not self.conn:
                self.connect()
            cursor = self.conn.cursor()
            for trade in trades:
                cursor.execute('''
                    INSERT INTO paper_trades (entry_price, exit_price, pnl, win, trade_size)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    trade.get("entry_price", 0), trade.get("exit_price", 0), trade.get("pnl", 0),
                    1 if trade.get("pnl", 0) > 0 else 0, trade.get("size", 0),
                ))
            self.conn.commit()
        except Exception as e:
            log.error("Error saving paper trades: %s", e)

    def save_backtest_results(self, results: dict):
        try:
            if not self.conn:
                self.connect()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO backtest_results
                (total_trades, win_rate, total_return, sharpe_ratio,
                 profit_factor, max_drawdown, recovery_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                results.get("total_trades", 0), results.get("win_rate", 0), results.get("total_return", 0),
                results.get("sharpe_ratio", 0), results.get("profit_factor", 0),
                results.get("max_drawdown", 0), results.get("recovery_factor", 0),
            ))
            self.conn.commit()
        except Exception as e:
            log.error("Error saving backtest results: %s", e)

    def save_patterns(self, patterns: dict):
        try:
            if not self.conn:
                self.connect()
            cursor = self.conn.cursor()
            for pattern_name, pattern_data in patterns.items():
                cursor.execute('''
                    INSERT INTO patterns (pattern_name, success_rate, occurrences, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (
                    pattern_name, pattern_data.get("success_rate", 0),
                    pattern_data.get("count", 0), pattern_data.get("confidence", 0),
                ))
            self.conn.commit()
        except Exception as e:
            log.error("Error saving patterns: %s", e)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


def _load_json_safe(filename: str) -> dict:
    try:
        path = Path(filename)
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _sqlite_save_once(db: DataPersistence) -> None:
    # connect()/close() both happen HERE, inside this one function call —
    # this function only ever runs as the target of a single
    # asyncio.to_thread() call (see sqlite_saver_loop), so the
    # connection's entire lifecycle stays on whichever one OS thread
    # services that call. See DataPersistence.connect()'s docstring for
    # the SIGSEGV this fixes.
    db.connect()
    try:
        omlx = _load_json_safe("omlx_metrics_live.json")
        if omlx:
            db.save_omlx_metrics(omlx)

        system = _load_json_safe("system_metrics.json")
        if system:
            db.save_system_metrics(system)

        paper = _load_json_safe("paper_trading_summary.json")
        if paper and paper.get("recent_trades"):
            db.save_paper_trades(paper["recent_trades"])

        backtest = _load_json_safe("backtest_summary.json")
        if backtest:
            db.save_backtest_results(backtest)

        if omlx:
            patterns = omlx.get("patterns", {})
            if patterns:
                db.save_patterns(patterns)
    finally:
        db.close()


async def sqlite_saver_loop(interval_seconds: int = SQLITE_SAVE_INTERVAL_SECONDS) -> None:
    db = DataPersistence("ethladder_analytics.db")
    try:
        while True:
            try:
                if cpu_limiter.check_and_throttle("sqlite_save"):
                    await asyncio.to_thread(_sqlite_save_once, db)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[sqlite_saver] save failed")
            await asyncio.sleep(interval_seconds)
    finally:
        db.close()


# =============================================================================
# Resource monitor (was export_system_metrics.py)
#
# Role in the bot: none directly — this feeds the dashboard's System tab
# and cpu_limiter's throttling decisions (see get_cpu_percent, which
# shells out to `top` independently of this loop's own sampling). Exists
# because the whole reason this file merges everything into one process
# is the CPU crash this system caused before — this is the visibility
# that would have caught it sooner.
# =============================================================================

async def resource_monitor_loop(interval_seconds: int = RESOURCE_MONITOR_INTERVAL_SECONDS) -> None:
    monitor = SystemResourceMonitor()
    while True:
        try:
            await asyncio.to_thread(monitor.record_metrics)
            await asyncio.to_thread(monitor.export_metrics, "system_metrics.json")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[resource_monitor] export failed")
        await asyncio.sleep(interval_seconds)


# =============================================================================
# Data pruner (was data_pruner.py / scheduled_pruner.py) — keeps the
# SQLite DB and disk usage bounded over long unattended runs.
# =============================================================================

def _prune_once(db_path: str = "ethladder_analytics.db") -> None:
    conn = None
    try:
        conn = __import__("sqlite3").connect(db_path)
        cursor = conn.cursor()

        threshold = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute("DELETE FROM omlx_metrics WHERE timestamp < ?", (threshold,))
        omlx_removed = cursor.rowcount
        cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (threshold,))
        system_removed = cursor.rowcount
        pattern_threshold = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("DELETE FROM patterns WHERE timestamp < ?", (pattern_threshold,))
        patterns_removed = cursor.rowcount
        conn.commit()
        if omlx_removed or system_removed or patterns_removed:
            log.info("[pruner] removed %d omlx / %d system / %d pattern rows older than retention window",
                      omlx_removed, system_removed, patterns_removed)

        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        if free_gb < 5:
            log.warning("[pruner] low disk space (%.1fGB free) — aggressive cleanup", free_gb)
            week_threshold = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (week_threshold,))
            month_threshold = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute("DELETE FROM paper_trades WHERE timestamp < ?", (month_threshold,))
            conn.commit()

        cursor.execute("VACUUM")
        conn.commit()
        log.info("[pruner] disk free: %.1fGB, db: %s", free_gb, db_path)
    finally:
        if conn is not None:
            conn.close()


async def pruner_loop(interval_seconds: int = PRUNER_INTERVAL_SECONDS) -> None:
    while True:
        try:
            await asyncio.to_thread(_prune_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[pruner] cycle failed")
        await asyncio.sleep(interval_seconds)


# =============================================================================
# Log watcher (was scripts/log_watcher.py) — tails THIS process's own log
# for ERROR/CRITICAL/WARNING lines and Python tracebacks and records them
# (SQLite + a plain-text fallback log) the moment something significant
# shows up. Previously also fired a native macOS desktop notification per
# alert — removed on request (was popping up for every ERROR/CRITICAL,
# including a 2332-in-90-seconds spam episode from routine backtest
# activity before that specific cause was separately fixed). Check
# logs/watcher_alerts.log or data/watcher_alerts.db to see what it caught.
# =============================================================================

ALERT_LOG_PATH = REPO_ROOT / "logs" / "watcher_alerts.log"
WATCHER_POLL_INTERVAL_SEC = 2.0
# A burst of identical repeated messages (e.g. a transient network blip
# retried every tick) records ONE alert, then goes quiet for this long
# before the same message can alert again.
WATCHER_DEDUP_WINDOW_SEC = 300.0
WATCHER_MAX_TRACEBACK_LINES = 40

_TIMESTAMPED_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_LEVEL_PATTERN = re.compile(r"\b(CRITICAL|ERROR|WARNING)\b")
_UVICORN_NATIVE_LEVEL = re.compile(r"^(CRITICAL|ERROR|WARNING):")


def classify_line(line: str) -> Optional[str]:
    """Returns a severity label if this line alone indicates something
    worth alerting on, else None."""
    if not line.strip():
        return None
    m = _UVICORN_NATIVE_LEVEL.match(line)
    if m:
        return m.group(1)
    m = _LEVEL_PATTERN.search(line)
    if m:
        return m.group(1)
    if "Traceback (most recent call last)" in line:
        return "ERROR"
    return None


def is_traceback_continuation(line: str) -> bool:
    """True for a line that's part of an in-progress traceback block
    rather than the start of an unrelated new log entry."""
    if not line.strip():
        return True
    if _TIMESTAMPED_LINE.match(line) or _UVICORN_NATIVE_LEVEL.match(line):
        return False
    return True


def append_alert_log(source: str, severity: str, text: str) -> None:
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG_PATH.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{source}] [{severity}] {text}\n")


def persist_alert(source: str, severity: str, text: str) -> None:
    """Best-effort SQLite write — a DB hiccup must never stop the watcher
    from still notifying/logging the alert via the other two channels."""
    try:
        conn = connect_watcher_db()
        try:
            record_alert(conn, source=source, severity=severity, message=text)
        finally:
            conn.close()
    except Exception:
        log.exception("[log_watcher] failed to persist alert to SQLite")


class Deduper:
    # Real leak this caps: _last_seen never evicted a key once added — a
    # dedup key is f"{source}:{severity}:{text[:200]}" (see watch_file),
    # and alert text (tracebacks especially) varies enough call to call
    # that distinct keys accumulate indefinitely over a long-running
    # process. This instance lives for the whole process (log_watcher_loop
    # creates it once). A simple size cap is enough here — dedup only
    # needs to remember the recent past, not forever.
    MAX_TRACKED_KEYS = 2_000

    def __init__(self, window_sec: float = WATCHER_DEDUP_WINDOW_SEC):
        self._window_sec = window_sec
        self._last_seen: dict[str, float] = {}

    def should_alert(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_seen.get(key)
        self._last_seen[key] = now
        if len(self._last_seen) > self.MAX_TRACKED_KEYS:
            # Evict the oldest half — cheap, and correctness only
            # requires "recent enough", not a precise LRU.
            oldest = sorted(self._last_seen.items(), key=lambda kv: kv[1])[:self.MAX_TRACKED_KEYS // 2]
            for stale_key, _ in oldest:
                del self._last_seen[stale_key]
        return last is None or (now - last) >= self._window_sec


def split_into_alert_blocks(lines: list[str]) -> list[tuple[str, str, str]]:
    """Scans a batch of newly-read lines and returns one (severity,
    first_line, text) tuple per alert-worthy block, collapsing a
    multi-line traceback into a single block rather than one entry per
    line."""
    blocks: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        severity = classify_line(lines[i])
        if severity is None:
            i += 1
            continue

        block = [lines[i]]
        j = i + 1
        while j < len(lines) and is_traceback_continuation(lines[j]) and len(block) < WATCHER_MAX_TRACEBACK_LINES:
            if not lines[j].strip() and j + 1 < len(lines) and not is_traceback_continuation(lines[j + 1]):
                break
            # A bare "Traceback (most recent call last):" line has no
            # timestamp, so is_traceback_continuation() alone can't tell
            # it apart from a genuine continuation line — two back-to-
            # back tracebacks would otherwise get merged into one block.
            # j == i + 1 is exempted: that's this block's OWN header
            # line, immediately following its triggering log line.
            if j > i + 1 and "Traceback (most recent call last)" in lines[j]:
                break
            block.append(lines[j])
            j += 1
        text = "\n".join(block).strip()
        first_line = block[0].strip()
        i = j if j > i else i + 1
        blocks.append((severity, first_line, text))
    return blocks


async def watch_file(path: Path, source: str, dedup: Deduper) -> None:
    log.info("[log_watcher] watching %s: %s", source, path)
    position = path.stat().st_size if path.exists() else 0

    while True:
        await asyncio.sleep(WATCHER_POLL_INTERVAL_SEC)
        if not path.exists():
            continue
        size = path.stat().st_size
        if size < position:
            position = 0  # truncated/rotated — start over from the beginning
        if size == position:
            continue

        with path.open("r", errors="replace") as f:
            f.seek(position)
            new_text = f.read()
            position = f.tell()

        for severity, first_line, text in split_into_alert_blocks(new_text.splitlines()):
            key = f"{source}:{severity}:{text[:200]}"
            if dedup.should_alert(key):
                append_alert_log(source, severity, text)
                persist_alert(source, severity, text)


async def log_watcher_loop() -> None:
    """Watches THIS process's one log file. Its own alert-emission calls
    above (append_alert_log / persist_alert) deliberately do NOT go
    through the "run_everything" logger — if they did, an alert about an
    error would itself land back in logs/run_everything.log, get read as
    new content on the next poll, and re-alert on itself forever
    (observed directly while building this: logs/uvicorn.log hit 110MB
    in under a minute under the old split-file design)."""
    await watch_file(REPO_ROOT / "logs" / "run_everything.log", "unified", Deduper())


# =============================================================================
# Inbox pattern extractor (symbot_python/exchange/inbox_pattern_extractor.py)
#
# Role in the bot: bridges the SEPARATE, external training-inbox
# pipeline (a different project entirely — see that module's docstring)
# into OMLX by reviewing its already-approved Q&A drafts for genuine,
# checkable DCA-strategy ideas, stored as PENDING candidates for a human
# to review/promote on the dashboard. Same hourly cadence as the source
# pipeline's own launchd job (com.ethladder.training-inbox,
# StartInterval=3600) — no reason to check more often than new approved
# drafts could plausibly exist. Shells out to the `claude` CLI
# (synchronous subprocess calls), so this runs via asyncio.to_thread()
# like every other CPU/IO-bound background task in this file — never
# directly on the shared event loop.
# =============================================================================

INBOX_EXTRACTOR_INTERVAL_SECONDS = 3600


async def inbox_pattern_extractor_loop(interval_seconds: int = INBOX_EXTRACTOR_INTERVAL_SECONDS) -> None:
    while True:
        try:
            added = await asyncio.to_thread(inbox_pattern_extractor.extract_new_candidates)
            if added:
                log.info("[inbox_pattern_extractor] found %d new candidate pattern(s)", added)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[inbox_pattern_extractor] cycle failed")
        await asyncio.sleep(interval_seconds)


# =============================================================================
# One combined FastAPI lifespan — starts every background task above in
# the same event loop uvicorn is already running, replacing the app's own
# (much smaller) lifespan.
# =============================================================================

@asynccontextmanager
async def unified_lifespan(app):
    request_optimizer_run()

    background = {
        "optimizer": asyncio.create_task(optimizer_loop(), name="optimizer"),
        "forward_tester": asyncio.create_task(forward_tester_loop(), name="forward_tester"),
        "ml_trainer": asyncio.create_task(ml_trainer_loop(), name="ml_trainer"),
        "data_exporter": asyncio.create_task(data_exporter_loop(), name="data_exporter"),
        "omlx_metrics_exporter": asyncio.create_task(omlx_metrics_exporter_loop(), name="omlx_metrics_exporter"),
        "sqlite_saver": asyncio.create_task(sqlite_saver_loop(), name="sqlite_saver"),
        "resource_monitor": asyncio.create_task(resource_monitor_loop(), name="resource_monitor"),
        "pruner": asyncio.create_task(pruner_loop(), name="pruner"),
        "log_watcher": asyncio.create_task(log_watcher_loop(), name="log_watcher"),
        "inbox_pattern_extractor": asyncio.create_task(inbox_pattern_extractor_loop(), name="inbox_pattern_extractor"),
    }
    log.info("=" * 80)
    log.info("ETH LADDER — UNIFIED SYSTEM STARTED (one process, one PID)")
    log.info("  web server + dashboard : http://127.0.0.1:8731/console")
    log.info("  trading bot (paper)    : in-process, symbot_python/api/paper.py")
    log.info("  background tasks       : %s", ", ".join(background))
    log.info("=" * 80)

    try:
        yield
    finally:
        log.info("Shutting down unified system — stopping %d background tasks...", len(background))
        for task in background.values():
            task.cancel()
        await asyncio.gather(*background.values(), return_exceptions=True)
        await trading_module.shutdown_manager()

        # Real bug this fixes: two SIGSEGV crashes (confirmed via macOS
        # crash reports — EXC_BAD_ACCESS, faulting thread
        # "ThreadPoolExecutor-N", top frame _pysqlite_query_execute)
        # both happened at THIS exact moment — right after "Shutting
        # down unified system" logged, during a restart. cancel() on a
        # task awaiting asyncio.to_thread() (sqlite_saver_loop,
        # optimizer_loop, etc.) cannot preemptively stop work already
        # running on a worker thread — the gather() above does wait for
        # those futures to resolve, but nothing explicitly waits for the
        # event loop's default ThreadPoolExecutor itself to fully join
        # before this lifespan (and therefore uvicorn's shutdown, and
        # then interpreter teardown) proceeds. shutdown_default_executor()
        # is the documented, explicit way to guarantee every worker
        # thread has actually finished — including any C-extension call
        # (sqlite3, xgboost, scikit-learn) that happened to be mid-flight
        # — before anything downstream can start tearing down objects
        # that thread might still be touching.
        await asyncio.get_running_loop().shutdown_default_executor()
        log.info("Unified system stopped.")


def main() -> None:
    # Swap the app's own lifespan for ours — same trick as re-registering
    # a Starlette Router's lifespan_context after construction. Done here
    # in main(), not at module import time: this module is also imported
    # by the test suite for its reusable functions/constants (SEARCH_GRID,
    # run_optimizer_cycle, classify_line, ...), and several other tests
    # (test_paper_routes.py) use `with TestClient(app_module.app)`, which
    # DOES trigger the ASGI lifespan — mutating the shared app's lifespan
    # as an import-time side effect would make an unrelated unit test
    # accidentally start all 9 real background tasks (real Bybit calls
    # included) the moment both modules happened to be imported in the
    # same pytest session.
    fastapi_app.router.lifespan_context = unified_lifespan
    # timeout_graceful_shutdown: give unified_lifespan's shutdown_default_executor()
    # (see its own comment) real time to actually join every worker
    # thread — uvicorn's own default graceful-shutdown window can be
    # short, and force-proceeding past it while a to_thread() call is
    # still mid-flight is the exact race that caused two confirmed
    # SIGSEGV crashes during restarts.
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8731, log_config=None, timeout_graceful_shutdown=30)


if __name__ == "__main__":
    main()
