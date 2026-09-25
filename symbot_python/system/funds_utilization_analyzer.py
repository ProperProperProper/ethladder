"""Automated funds utilization percentage analyzer.

Tests the current best-performing strategy across all utilization percentages
(78% to 100%) in 1% increments, runs walk-forward backtests on each, and
identifies the optimal percentage based on profit factor and risk metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from symbot_python.signals.candles import bars_for_days, DEFAULT_BACKTEST_DAYS
from symbot_python.strategy.backtest import run_backtest, BACKTEST_ARGS
from symbot_python.strategy.models import BotConfig, to_exchange_symbol
from symbot_python.strategy.optimization_store import connect, get_best_current_winner
from symbot_python.exchange.factory import create_exchange_client
from symbot_python.exchange.base import TradingMode

logger = logging.getLogger(__name__)

SYMBOL = "ETHUSDT"
PAIR = "ETH/USDT"
MIN_UTILIZATION = 78
MAX_UTILIZATION = 100


@dataclass
class UtilizationResult:
    """Result for one utilization percentage test."""
    utilization_percent: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    total_pnl_pct: float
    is_optimal: bool = False


async def analyze_all_utilizations() -> dict:
    """Test all utilization percentages (78-100%) and return ranked results."""
    try:
        # Get current best winner parameters
        db_path = Path("ethladder_analytics.db")
        if not db_path.exists():
            return {"error": "No database found", "results": []}

        conn = connect(db_path)
        winner = get_best_current_winner(conn, SYMBOL)
        if not winner:
            logger.warning("No winner found; using fallback params")
            winner = {}

        # Get historical candles
        client = await create_exchange_client(TradingMode.PAPER)
        candles = await client.get_kline(SYMBOL, "1D", bars_for_days(DEFAULT_BACKTEST_DAYS))
        if not candles:
            return {"error": "No candle data", "results": []}

        results: list[UtilizationResult] = []

        # Test each utilization percentage
        for util_pct in range(MIN_UTILIZATION, MAX_UTILIZATION + 1):
            try:
                # Build config with this utilization
                config = BotConfig(
                    pair=PAIR,
                    side=winner.get("side", "long"),
                    leverage=winner.get("leverage", 10.0),
                    dca_order_step_percent=winner.get("dca_order_step_percent", 1.3),
                    dca_order_size_multiplier=winner.get("dca_order_size_multiplier", 1.08),
                    dca_max_order=winner.get("dca_max_order", 10),
                    dca_order_step_percent_multiplier=winner.get("dca_order_step_percent_multiplier", 1.0),
                    dca_take_profit_percent=winner.get("dca_take_profit_percent", 0.33),
                    exchange_fee=winner.get("exchange_fee", 0.06),
                    dca_stop_loss_enabled=winner.get("dca_stop_loss_enabled", False),
                    dca_stop_loss_percent=winner.get("dca_stop_loss_percent", 10.0),
                    dca_trailing_stop_enabled=winner.get("dca_trailing_stop_enabled", False),
                    dca_trailing_stop_distance=winner.get("dca_trailing_stop_distance", 1.0),
                    dca_trailing_activate_profit=winner.get("dca_trailing_activate_profit", 1.0),
                    reverse_drawdown_percent=winner.get("reverse_drawdown_percent", 0.0),
                    reverse_cooldown_sec=winner.get("reverse_cooldown_sec", 3600),
                    max_consecutive_reversals=winner.get("max_consecutive_reversals", 0),
                    funds_utilization_percent=float(util_pct),
                )

                # Run backtest
                bt_result = await run_backtest(
                    candles=candles,
                    config=config,
                    **BACKTEST_ARGS,
                )

                result = UtilizationResult(
                    utilization_percent=util_pct,
                    profit_factor=bt_result.profit_factor,
                    win_rate=bt_result.win_rate * 100,
                    max_drawdown=bt_result.max_drawdown_pct,
                    total_trades=bt_result.total_trades,
                    total_pnl_pct=bt_result.total_pnl_pct,
                )
                results.append(result)
                logger.info(f"  {util_pct}%: PF={result.profit_factor:.2f}, WR={result.win_rate:.1f}%, DD={result.max_drawdown:.1f}%")

            except Exception as e:
                logger.warning(f"Error testing {util_pct}%: {e}")

        # Rank by profit factor (primary) then by max drawdown (secondary)
        if results:
            best = max(results, key=lambda r: (r.profit_factor, -r.max_drawdown))
            best.is_optimal = True
            results_sorted = sorted(results, key=lambda r: (r.profit_factor, -r.max_drawdown), reverse=True)
            return {
                "timestamp": str(Path("ethladder_analytics.db").stat().st_mtime),
                "best_utilization": best.utilization_percent,
                "results": [
                    {
                        "utilization_percent": r.utilization_percent,
                        "profit_factor": round(r.profit_factor, 3),
                        "win_rate": round(r.win_rate, 1),
                        "max_drawdown": round(r.max_drawdown, 2),
                        "total_trades": r.total_trades,
                        "total_pnl_pct": round(r.total_pnl_pct, 2),
                        "is_optimal": r.is_optimal,
                    }
                    for r in results_sorted
                ],
            }
        else:
            return {"error": "No results generated", "results": []}

    except Exception as e:
        logger.exception("Funds utilization analysis failed: %s", e)
        return {"error": str(e), "results": []}
