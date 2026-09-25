"""Live trading control surface — real Bybit mainnet, real money at risk.
Mirrors paper trading exactly (same DCA logic, same entry/exit rules) but
places actual orders on Bybit with real USDT balance from your account.

🚨 REAL MONEY WARNING 🚨
- Uses REAL Bybit account balance (read from Keychain)
- Places REAL orders with REAL funds on mainnet
- Leverage: 9-11x on ETHUSDT
- NO simulated fills — all orders execute on Bybit mainnet

Bots/deals live in-process only; restarting the server does NOT clear state —
positions remain open on Bybit and must be managed manually if needed.

NEVER places reverse trades (opposite direction). If paper places a LONG
entry, live ALSO places a LONG entry. No hedging, no reversals.

Live trading always runs the continuous optimizer's current best winner
for ETHUSDT, auto-refreshed every PARAM_REFRESH_SECONDS (3x/day).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from symbot_python.api.equity import EquityHistory
from symbot_python.exchange.base import TradingMode, ExchangeClient
from symbot_python.exchange.factory import create_exchange_client
from symbot_python.logging_setup import MELBOURNE_TZ
from symbot_python.exchange.keychain import fetch_real_balance
from symbot_python.strategy.dca_bot_manager import DCABotManager
from symbot_python.strategy.dca_math import calculate_liquidation_price
from symbot_python.strategy.backtest import FIXED_TAKE_PROFIT_PERCENT
from symbot_python.strategy.leverage_policy import DEFAULT_LEVERAGE, clamp_leverage
from symbot_python.strategy.funds_utilization_policy import (
    DEFAULT_FUNDS_UTILIZATION_PERCENT,
    clamp_funds_utilization,
)
from symbot_python.strategy.models import BotConfig, DealStatus, to_exchange_symbol
from symbot_python.strategy.optimization_store import connect, get_best_current_winner, row_params

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

# Locked, not configurable — same policy as the backtester's ETHUSDT lock.
# Leverage itself is a bounded band (9x-11x), not a single fixed value —
# see strategy/leverage_policy.py. See TESTING_POLICY.md / [[feedback_non_negotiables]].
SYMBOL = "ETHUSDT"
PAIR = "ETH/USDT"
AUTO_BOT_NAME = f"live-auto-{SYMBOL}"
PARAM_REFRESH_SECONDS = 24 * 3600 // 3  # "params should update in live trading 3 times a day"

# Fields the continuous optimizer actually searches (see SEARCH_GRID in
# scripts/continuous_optimizer.py) — everything else (order sizing,
# fee %, trailing stop) isn't searched, so it keeps a sane fixed default.
WINNER_PARAM_FIELDS = (
    "dca_order_step_percent", "dca_order_size_multiplier",
    "dca_max_order", "dca_order_step_percent_multiplier", "dca_stop_loss_enabled",
    "dca_stop_loss_percent", "dca_trailing_stop_enabled", "dca_trailing_stop_distance",
    "dca_trailing_activate_profit", "reverse_drawdown_percent", "reverse_cooldown_sec",
    "max_consecutive_reversals", "side", "leverage", "funds_utilization_percent",
)

FALLBACK_BOT_DEFAULTS = {
    "side": "long",
    "leverage": DEFAULT_LEVERAGE,
    "first_order_amount": 20.0,
    "dca_order_amount": 45.0,
    "dca_max_order": 10,
    "dca_order_size_multiplier": 1.08,
    "dca_order_start_distance": 1.3,
    "dca_order_step_percent": 1.3,
    "dca_order_step_percent_multiplier": 1.0,
    "dca_take_profit_percent": FIXED_TAKE_PROFIT_PERCENT,
    "exchange_fee": 0.06,
    "dca_stop_loss_enabled": False,
    "dca_stop_loss_percent": 10.0,
    "dca_trailing_stop_enabled": False,
    "dca_trailing_stop_distance": 1.0,
    "dca_trailing_activate_profit": 1.0,
    "auto_size_to_funds": True,
    # Fallback-only now: funds_utilization_percent is in
    # WINNER_PARAM_FIELDS, so once a real winner exists this gets
    # overridden by whatever the continuous optimizer's search (which
    # includes this dimension — see SEARCH_GRID) found actually
    # maximizes PnL for the real risk involved, not a guessed constant.
    # This 85% only applies before any winner has ever been recorded.
    # (98% used to be hardcoded everywhere and left zero headroom for a
    # fill landing worse than its sizing-time price estimate — see
    # dca_bot.py's tick-loop optimism-gap note — which drove the paper
    # account negative once a fully-deployed ladder overshot slightly.)
    "funds_utilization_percent": DEFAULT_FUNDS_UTILIZATION_PERCENT,
}

_manager: Optional[DCABotManager] = None
_refresh_task: Optional[asyncio.Task] = None
_equity_task: Optional[asyncio.Task] = None
_equity_history: Optional[EquityHistory] = None
_last_param_sync: Optional[dict] = None  # for display: where the current params came from
_trading_enabled: bool = False  # Manual start/stop control
# Session telemetry baseline — set once, the moment live trading starts
# with your real Bybit available balance. Everything on the dashboard
# (session P/L %, the equity curve) is measured from this point.
# Restarting the server starts a new session baseline but DOES NOT close
# existing positions on Bybit — they remain open.
_session_start_balance: Optional[float] = None
_session_start_time: Optional[float] = None



# Paper trading is entirely simulated (PaperExchangeClient can never reach
# an authenticated Bybit endpoint — see its own module docstring), so a
# fallback starting balance here risks nothing real. Only used when the
# real account's available balance is effectively 0 (funds tied up
# elsewhere, account not yet funded, etc.) — a ~$0 paper wallet can never
# open a single deal, which defeats the entire point of a paper-trading
# page: there is nothing to test with, forever, until the real account
# changes. $200 is enough headroom for the DCA ladder's default sizing to
# actually place trades.
FALLBACK_STARTING_BALANCE_USDT = 200.0
# "Effectively 0", not exactly 0: a real account can carry sub-cent dust
# (e.g. $0.0000390 observed directly here — fractional leftovers from
# fees/funding) that displays as "0.0000" at the dashboard's 4-decimal
# precision but is technically > 0, so a plain `<= 0` check never fires.
# $1 is comfortably below anything a real, usable balance would be, and
# comfortably above any dust amount.
MIN_USABLE_BALANCE_USDT = 1.0


async def get_manager() -> DCABotManager:
    """Live trading uses your REAL available Bybit balance from Keychain
    (read-only lookup, same as the continuous optimizer). Fetched once at
    startup; the live bot then places real orders on Bybit mainnet from
    that starting point.

    🚨 REAL MONEY — positions remain open even if the server restarts.
    """
    global _manager, _refresh_task, _session_start_balance, _session_start_time, _equity_task, _equity_history
    if _manager is None:
        balance = await fetch_real_balance()
        starting_balance = balance.total_available_balance
        if starting_balance < MIN_USABLE_BALANCE_USDT:
            logger.error(
                "🚨 LIVE TRADING BLOCKED 🚨: Real available balance is %.8f USDT — "
                "insufficient funds to open a position. Minimum required: %.2f USDT",
                starting_balance, MIN_USABLE_BALANCE_USDT,
            )
            raise ValueError(
                f"Insufficient balance for live trading: {starting_balance:.8f} USDT "
                f"(minimum: {MIN_USABLE_BALANCE_USDT:.2f} USDT)"
            )
        logger.warning("🚨 LIVE TRADING ACTIVE 🚨 Using real Bybit balance: %.2f USDT", starting_balance)
        client = await create_exchange_client(TradingMode.LIVE)
        if not isinstance(client, ExchangeClient):
            raise TypeError("Live manager requires an authenticated exchange client")
        _manager = DCABotManager(client)
        _session_start_balance = starting_balance
        _session_start_time = time.time()
        _equity_history = EquityHistory(_session_start_time, _session_start_balance)
        await _manager.start()
        await _sync_bot_with_latest_winner(_manager, force_create=True)
        _refresh_task = asyncio.create_task(_auto_refresh_loop(_manager))
        _equity_task = asyncio.create_task(_sample_equity_loop(_manager, _equity_history))
    return _manager


async def shutdown_manager() -> None:
    global _manager, _refresh_task, _equity_task
    if _equity_task is not None:
        _equity_task.cancel()
        await asyncio.gather(_equity_task, return_exceptions=True)
        _equity_task = None
    if _refresh_task is not None:
        _refresh_task.cancel()
        _refresh_task = None
    if _manager is not None:
        close = getattr(_manager.exchange, "close", None)
        if close is not None:
            # Bybit client.close() closes websocket connections
            await asyncio.to_thread(close)
        await _manager.stop()
        _manager = None


async def check_open_positions() -> bool:
    """Check if there are open positions on Bybit. Auto-start trading if positions exist."""
    global _trading_enabled
    try:
        client = await create_exchange_client(TradingMode.LIVE)
        position = client.position(SYMBOL)
        if position and position.qty != 0:
            logger.info("🚨 Found open position on Bybit: %.4f %s at %s. Auto-enabling live trading.",
                       position.qty, SYMBOL, position.avg_price)
            _trading_enabled = True
            return True
    except Exception as e:
        logger.warning("Could not check for open positions: %s", e)
    return False


async def start_live_trading() -> dict:
    """Start live trading. Only places orders if _trading_enabled is True."""
    global _trading_enabled, _manager
    _trading_enabled = True
    if _manager is None:
        try:
            await get_manager()
            return {"status": "started", "message": "Live trading started"}
        except Exception as e:
            _trading_enabled = False
            return {"status": "error", "message": str(e)}
    return {"status": "already_running", "message": "Live trading already running"}


async def stop_live_trading() -> dict:
    """Stop live trading. Positions remain open on Bybit."""
    global _trading_enabled
    _trading_enabled = False
    return {"status": "stopped", "message": "Live trading stopped. Existing positions remain open on Bybit."}


async def close_all_positions() -> dict:
    """Close all open positions on Bybit immediately."""
    global _manager
    if _manager is None:
        await get_manager()

    try:
        position = _manager.exchange.position(SYMBOL)
        if position and position.qty != 0:
            # Close by placing opposite order
            side = "sell" if position.qty > 0 else "buy"
            logger.warning("🔴 CLOSING ALL POSITIONS 🔴 — Placing %s order for %.4f %s",
                          side, abs(position.qty), SYMBOL)
            await _manager.exchange.place_market_order(
                symbol=SYMBOL,
                side=side,
                qty=abs(position.qty),
                reduce_only=True
            )
            return {"status": "success", "message": f"Closed {abs(position.qty)} {SYMBOL}"}
        else:
            return {"status": "no_position", "message": "No open position to close"}
    except Exception as e:
        logger.error("Failed to close positions: %s", e)
        return {"status": "error", "message": str(e)}


async def _sample_equity(manager: DCABotManager, history: EquityHistory) -> None:
    # Read actual simulated fills and committed margin, not planned ladder sizes.
    ticker = await manager.exchange.get_ticker(SYMBOL)
    position = manager.exchange.position(SYMBOL)
    funding = sum(d.funding_cost_quote for d in manager.deals.values()
                  if d.status == DealStatus.ACTIVE)
    # Booked net P/L includes paid entry/exit fees and settled funding.
    # Unsettled funding belongs to the still-open position until settlement.
    realised = manager.exchange.margin_balance + position.margin_committed - history.start_balance
    unrealised = position.qty * (ticker.last - position.avg_price) - funding
    equity = history.start_balance + realised + unrealised
    history.record(time.time(), equity, realised, unrealised)


async def _sample_equity_loop(manager: DCABotManager, history: EquityHistory) -> None:
    while True:
        try:
            await _sample_equity(manager, history)
        except Exception:
            logger.warning("Equity sample unavailable; retrying in 60 seconds.", exc_info=True)
        await asyncio.sleep(60)


def _params_from_best_winner() -> Optional[dict]:
    conn = None
    try:
        conn = connect()
        row = get_best_current_winner(conn, SYMBOL)
    except Exception:
        # A DB error (e.g. "database is locked" under write contention
        # from the concurrently-running continuous optimizer) must fall
        # back to defaults, not crash whatever route called this.
        logger.exception("Failed to read the best current winner — falling back to defaults.")
        return None
    finally:
        if conn is not None:
            conn.close()
    if row is None:
        return None
    winner_params = row_params(row)
    config = dict(FALLBACK_BOT_DEFAULTS)
    for field in WINNER_PARAM_FIELDS:
        if field in winner_params:
            config[field] = winner_params[field]
    return {
        "config": config,
        "source_interval": row["interval"],
        "source_score": row["in_sample_score"],
        "source_tested_at": row["tested_at"],
    }


async def _sync_bot_with_latest_winner(manager: DCABotManager, force_create: bool = False) -> None:
    """Update future-deal settings from the current best winner.

    Existing positions retain their own configuration until closed. A
    direction change affects the next deal; it does not close or flip the
    active paper position.
    """
    global _last_param_sync
    result = _params_from_best_winner()
    config = dict(result["config"] if result else FALLBACK_BOT_DEFAULTS)
    # Defense in depth: clamp to [MIN_LEVERAGE, MAX_LEVERAGE] regardless
    # of what a stored winner row or fallback default says — leverage
    # must never leave this band no matter where the value came from.
    config["leverage"] = clamp_leverage(config.get("leverage", DEFAULT_LEVERAGE))
    # Same defense-in-depth, same reasoning: a deal sized below
    # MIN_FUNDS_UTILIZATION_PERCENT commits too little of the account for
    # even a fully-successful ladder to produce a meaningful profit — a
    # stored winner row from before this floor existed (or a manually
    # edited one) must never leave the bot running under-sized.
    config["funds_utilization_percent"] = clamp_funds_utilization(
        config.get("funds_utilization_percent", DEFAULT_FUNDS_UTILIZATION_PERCENT)
    )
    # Take profit is fixed across optimizer, paper, and live paths. Old
    # winner rows can contain a searched TP, but must never override it.
    config["dca_take_profit_percent"] = FIXED_TAKE_PROFIT_PERCENT

    bot = next((b for b in manager.bots.values() if b.bot_name == AUTO_BOT_NAME), None)
    if bot is None:
        if not force_create:
            return
        bot = BotConfig(bot_name=AUTO_BOT_NAME, pair=PAIR, **config)
        manager.add_bot(bot)
        try:
            symbol = to_exchange_symbol(bot.pair)
            await manager.exchange.get_precision(symbol)
            await manager.request_deal_start(bot.bot_id)
        except Exception:
            logger.exception("Failed to start initial auto paper deal for %s", AUTO_BOT_NAME)
    else:
        for field, value in config.items():
            setattr(bot, field, value)

    _last_param_sync = {
        "synced_at": time.time(),
        "source_interval": result["source_interval"] if result else None,
        "source_score": result["source_score"] if result else None,
        "source_tested_at": result["source_tested_at"] if result else None,
        "used_fallback": result is None,
    }


async def _auto_refresh_loop(manager: DCABotManager) -> None:
    """Runs forever: every PARAM_REFRESH_SECONDS, re-checks the param
    library for a newer/better winner and hot-applies it. A transient
    failure (e.g. the optimizer hasn't recorded anything yet, or the DB
    is mid-write) must never kill this loop — it just tries again next cycle.
    """
    try:
        while True:
            await asyncio.sleep(PARAM_REFRESH_SECONDS)
            try:
                await _sync_bot_with_latest_winner(manager)
                logger.info("Auto-refreshed paper bot params from param library.")
            except Exception:
                logger.exception("Param auto-refresh failed — will retry next cycle.")
    except asyncio.CancelledError:
        pass


def _format_melbourne(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, MELBOURNE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


async def _deal_view(manager: DCABotManager, bot: BotConfig, deal, current_price: Optional[float]) -> dict:
    bot = deal.config or bot
    last_filled = deal.orders[deal.filled_count - 1] if deal.filled_count else None
    unrealized_percent = None
    unrealized_quote = None
    notional = None
    margin_used = None
    liquidation_price = None

    if last_filled:
        margin_used = (last_filled.qty_sum * last_filled.average) / bot.leverage if bot.leverage > 0 else None

    if deal.status == DealStatus.ACTIVE and deal.is_start == 1 and last_filled and current_price:
        sign = -1 if bot.side == "short" else 1
        unrealized_percent = sign * (current_price - last_filled.average) / last_filled.average * 100
        unrealized_quote = sign * last_filled.qty_sum * (current_price - last_filled.average)
        notional = last_filled.qty_sum * current_price
        if bot.leverage > 1:
            try:
                symbol = to_exchange_symbol(bot.pair)
                mmr = await manager.exchange.get_maintenance_margin_rate(symbol)
                liquidation_price = calculate_liquidation_price(
                    last_filled.average, bot.leverage, mmr, side=bot.side
                )
            except Exception:
                liquidation_price = None

    # Calculate entry/exit prices
    entry_price = deal.orders[0].average if deal.orders else None
    exit_price = deal.sell_data.get("close_price") if deal.sell_data else None

    # Calculate duration
    duration_seconds = None
    if deal.date_opened and deal.date_closed:
        duration_seconds = int(deal.date_closed - deal.date_opened)

    # Calculate liquidation distance
    liq_distance_percent = None
    if liquidation_price and last_filled and current_price:
        sign = -1 if bot.side == "short" else 1
        liq_distance_percent = abs(sign * (liquidation_price - current_price) / (liquidation_price - last_filled.average) * 100) if (liquidation_price - last_filled.average) != 0 else None

    # Margin percentage
    margin_percent = None
    if margin_used and current_price and last_filled:
        position_value = last_filled.qty_sum * current_price
        margin_percent = (margin_used / position_value * 100) if position_value > 0 else None

    return {
        "deal_id": deal.deal_id,
        "side": bot.side,
        "leverage": bot.leverage,
        "status": "closed" if deal.status == DealStatus.CLOSED else ("filling" if deal.is_start == 0 else "monitoring"),
        "paused": deal.paused,
        "pause_reason": deal.pause_reason,
        "filled_count": deal.filled_count,
        "rung_count": len(deal.orders),
        "qty": last_filled.qty_sum if last_filled else None,
        "entry_price": entry_price,
        "average": last_filled.average if last_filled else None,
        "exit_price": exit_price,
        "target": last_filled.target if last_filled else None,
        "unrealized_percent": unrealized_percent,
        "unrealized_quote": unrealized_quote,
        "notional": notional,
        "margin_used": margin_used,
        "margin_percent": margin_percent,
        "liquidation_price": liquidation_price,
        "liq_distance_percent": liq_distance_percent,
        "funding_cost_quote": deal.funding_cost_quote,
        "duration_seconds": duration_seconds,
        "sell_data": deal.sell_data,
        "date_opened": deal.date_opened,
        "opened_at": _format_melbourne(deal.date_opened),
        "closed_at": _format_melbourne(deal.date_closed),
        "stop_loss": deal.stop_loss,
        "liquidated": deal.liquidated,
        "panic_sell": deal.panic_sell,
        "canceled": deal.canceled,
        "config": {
            "dca_stop_loss_enabled": bot.dca_stop_loss_enabled,
            "dca_trailing_stop_enabled": bot.dca_trailing_stop_enabled,
            "reverse_drawdown_percent": bot.reverse_drawdown_percent,
        }
    }


def _svg_equity_curve(points: list[float], width: int = 640, height: int = 140) -> str:
    """Hand-rolled inline SVG rather than a JS charting library — this app
    has no other JS dependency, and a personal single-user tool shouldn't
    need one just for a sparkline. Pure server-side float formatting, no
    user-controlled text, so the caller can safely render it with |safe.
    """
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    step = width / (len(points) - 1)
    coords = " ".join(
        f"{i * step:.1f},{height - ((v - lo) / span) * height:.1f}" for i, v in enumerate(points)
    )
    baseline_y = height - ((points[0] - lo) / span) * height
    color = "#2e9e4f" if points[-1] >= points[0] else "#c93b3b"
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="width:100%; height:{height}px; background:#8881; border-radius:8px; display:block;">'
        f'<line x1="0" y1="{baseline_y:.1f}" x2="{width}" y2="{baseline_y:.1f}" '
        f'stroke="#8886" stroke-dasharray="4,4" />'
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" />'
        f'</svg>'
    )


def _session_summary(bots_view: list[dict], free_cash_balance: float) -> dict:
    """Full trade-level telemetry across every bot's deals: realized/
    unrealized P/L, win rate, profit factor, best/worst trade, a
    close-reason breakdown (tp/stop_loss/cancel/panic_sell/engine_error/
    liquidated), and a reconstructed equity curve (starting balance +
    cumulative realized P/L after each closed deal, in close order —
    there's no separate persisted balance time series, but this is
    exactly derivable since every close nets its P/L into the wallet
    once, at close, and nowhere else — see dca_bot.py's _handle_sell).

    `free_cash_balance` (manager.exchange.get_balance()'s USDT figure)
    is spendable margin ONLY — PaperExchangeClient debits margin_required
    from it the instant an order fills (see paper_client.py's
    place_market_order), so it excludes whatever's currently locked in
    any open deal's position. "Balance" below must instead be total
    account EQUITY (free cash + margin locked in open deals + their
    unrealized P/L) — otherwise, the moment a deal has any safety orders
    filled, "Balance" reads as a huge apparent loss against `start:`
    (which WAS free-cash-equals-equity, before anything was open) even
    though nothing has actually been lost, it's just locked as margin.
    """
    closed = sorted(
        (
            {**d, "bot_name": b["bot_name"]}
            for b in bots_view for d in b["deals"]
            if d["status"] == "closed" and d["sell_data"]
        ),
        key=lambda d: d["sell_data"]["date"],
    )
    open_deals = [d for b in bots_view for d in b["deals"] if d["status"] != "closed"]

    total_trades = len(closed)
    wins = [d for d in closed if d["sell_data"]["is_real_win"]]
    losses = [d for d in closed if not d["sell_data"]["is_real_win"]]
    realized_pnl_quote = sum(d["sell_data"]["profit_quote"] for d in closed)
    unrealized_pnl_quote = sum(
        d["unrealized_quote"] for d in open_deals if d["unrealized_quote"] is not None
    )
    margin_locked = sum(
        d["margin_used"] for d in open_deals if d["margin_used"] is not None
    )
    current_balance = free_cash_balance + margin_locked + unrealized_pnl_quote

    gross_win = sum(d["sell_data"]["profit_quote"] for d in wins)
    gross_loss = sum(d["sell_data"]["profit_quote"] for d in losses)

    reason_counts: dict[str, int] = {}
    for d in closed:
        reason = d["sell_data"]["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    equity_curve = None
    max_drawdown_percent = None
    if _session_start_balance is not None and closed:
        running = _session_start_balance
        curve = [running]
        peak = running
        max_dd = 0.0
        for d in closed:
            running += d["sell_data"]["profit_quote"]
            curve.append(running)
            peak = max(peak, running)
            if peak > 0:
                max_dd = max(max_dd, (peak - running) / peak * 100)
        equity_curve = curve
        max_drawdown_percent = max_dd

    best_trade = max(closed, key=lambda d: d["sell_data"]["profit_quote"], default=None)
    worst_trade = min(closed, key=lambda d: d["sell_data"]["profit_quote"], default=None)

    return {
        "started_at": _format_melbourne(_session_start_time),
        "start_balance": _session_start_balance,
        "current_balance": current_balance,
        "realized_pnl_quote": realized_pnl_quote,
        "realized_pnl_percent": (
            realized_pnl_quote / _session_start_balance * 100
            if _session_start_balance else None
        ),
        "unrealized_pnl_quote": unrealized_pnl_quote,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_percent": (len(wins) / total_trades * 100) if total_trades else None,
        "avg_win_quote": (gross_win / len(wins)) if wins else None,
        "avg_loss_quote": (gross_loss / len(losses)) if losses else None,
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss < 0 else None,
        "best_trade_quote": best_trade["sell_data"]["profit_quote"] if best_trade else None,
        "worst_trade_quote": worst_trade["sell_data"]["profit_quote"] if worst_trade else None,
        "max_drawdown_percent": max_drawdown_percent,
        "reason_counts": reason_counts,
        "equity_curve": equity_curve or [],
        "equity_svg": _svg_equity_curve(equity_curve) if equity_curve else "",
        "open_deal_count": len(open_deals),
    }


async def _view_context(manager: DCABotManager, error: Optional[str]) -> dict:
    bots_view = []
    for bot in manager.bots.values():
        deals = sorted(
            (d for d in manager.deals.values() if d.bot_id == bot.bot_id),
            key=lambda d: d.date_opened, reverse=True,
        )
        current_price = None
        try:
            symbol = to_exchange_symbol(bot.pair)
            ticker = await manager.exchange.get_ticker(symbol)
            current_price = ticker.last
        except Exception:
            current_price = None

        deal_views = [await _deal_view(manager, bot, d, current_price) for d in deals]
        bots_view.append({
            "bot_id": bot.bot_id, "bot_name": bot.bot_name, "pair": bot.pair,
            "active": bot.active, "deal_count": bot.deal_count,
            "current_price": current_price, "deals": deal_views,
            "leverage": bot.leverage, "side": bot.side,
            "dca_take_profit_percent": bot.dca_take_profit_percent,
            "dca_order_step_percent": bot.dca_order_step_percent,
            "dca_order_size_multiplier": bot.dca_order_size_multiplier,
            "dca_max_order": bot.dca_max_order,
            "dca_stop_loss_enabled": bot.dca_stop_loss_enabled,
        })

    balances = await manager.exchange.get_balance()
    current_balance = balances.get("USDT", 0.0)
    session = _session_summary(bots_view, current_balance)

    return {
        "error": error,
        "bots": bots_view,
        "balances": balances,
        "session": session,
        "circuit_breaker_active": manager.circuit_breaker_active,
        "param_sync": _last_param_sync,
        "param_refresh_hours": PARAM_REFRESH_SECONDS / 3600,
        "equity_chart": _equity_history.render() if _equity_history else "",
    }


@router.get("/live", response_class=HTMLResponse)
async def paper_page(request: Request):
    manager = await get_manager()
    context = await _view_context(manager, None)
    return templates.TemplateResponse(request, "paper.html", context)



@router.get("/live/equity", response_class=HTMLResponse)
async def paper_equity():
    await get_manager()
    return HTMLResponse(_equity_history.render() if _equity_history else "Collecting equity…")


@router.post("/live/update-params")
async def paper_update_params():
    """The 'Update to latest params now' button — immediately re-syncs
    the auto-managed bot with whatever the param library's best current
    winner is, instead of waiting for the next PARAM_REFRESH_SECONDS tick.
    """
    manager = await get_manager()
    await _sync_bot_with_latest_winner(manager, force_create=True)
    return RedirectResponse("/live", status_code=303)


@router.post("/live/clear-circuit-breaker")
async def paper_clear_circuit_breaker():
    """The circuit breaker (DCABotManager.circuit_breaker_active) has no
    automatic trip condition wired in today — nothing sets it, so it can
    never trip. This route exists as defense in depth for if/when
    something does: once tripped, there must be a way to clear it from
    the UI without restarting the whole server (which would also wipe
    all paper state), so "let it run unattended" doesn't quietly become
    "requires an SSH session to un-stick."
    """
    global _manager
    if _manager is not None:
        _manager.clear_circuit_breaker()
    return RedirectResponse("/live", status_code=303)


@router.post("/live/bots/{bot_id}/stop")
async def paper_stop_bot(bot_id: str):
    manager = await get_manager()
    bot = manager.bots.get(bot_id)
    if bot:
        bot.active = False  # stops future auto-chain; open deals keep running
    return RedirectResponse("/live", status_code=303)


@router.post("/live/deals/{deal_id}/cancel")
async def paper_cancel_deal(deal_id: str):
    manager = await get_manager()
    manager.cancel_deal(deal_id)
    return RedirectResponse("/live", status_code=303)


@router.post("/live/deals/{deal_id}/panic")
async def paper_panic_deal(deal_id: str):
    manager = await get_manager()
    manager.panic_sell_deal(deal_id)
    return RedirectResponse("/live", status_code=303)


@router.post("/live/start")
async def start_live_trading_route():
    """Start live trading (manual control)."""
    result = await start_live_trading()
    return result


@router.post("/live/stop")
async def stop_live_trading_route():
    """Stop live trading (positions remain open on Bybit)."""
    result = await stop_live_trading()
    return result


@router.post("/live/close-all-positions")
async def close_positions_route():
    """Close all open positions on Bybit immediately."""
    result = await close_all_positions()
    return result


@router.get("/live/status")
async def live_trading_status():
    """Get live trading status (enabled/disabled)."""
    global _trading_enabled, _manager
    manager_exists = _manager is not None
    position_data = None

    if manager_exists:
        try:
            position = _manager.exchange.position(SYMBOL)
            if position:
                position_data = {
                    "qty": position.qty,
                    "avg_price": position.avg_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "leverage": position.leverage,
                }
        except Exception:
            pass

    return {
        "trading_enabled": _trading_enabled,
        "manager_initialized": manager_exists,
        "position": position_data,
        "symbol": SYMBOL,
    }
