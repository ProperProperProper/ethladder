"""Unified trading system — ONE manager trades both paper and live simultaneously.

No mirroring, no separate managers, no bot_id mismatch. A single DCABotManager
instance executes every order on BOTH paper (simulated) and live (real Bybit)
in one atomic call via HybridExchangeClient.

Paper positions tracked in paper_client.
Live positions tracked via Bybit's actual position state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from symbot_python.exchange.base import TradingMode
from symbot_python.exchange.factory import create_exchange_client
from symbot_python.exchange.hybrid_client import HybridExchangeClient
from symbot_python.exchange.paper_client import PaperExchangeClient
from symbot_python.exchange.keychain import fetch_real_balance
from symbot_python.strategy.dca_bot_manager import DCABotManager
from symbot_python.strategy.optimization_store import get_best_current_winner, row_params
from symbot_python.strategy.models import BotConfig, to_exchange_symbol
from symbot_python.strategy.backtest import FIXED_TAKE_PROFIT_PERCENT
from symbot_python.strategy.leverage_policy import DEFAULT_LEVERAGE
from symbot_python.strategy.funds_utilization_policy import DEFAULT_FUNDS_UTILIZATION_PERCENT

logger = logging.getLogger(__name__)

SYMBOL = "ETHUSDT"
PAIR = "ETH/USDT"

_manager: Optional[DCABotManager] = None
_trading_enabled: bool = False
_param_sync_task: Optional[asyncio.Task] = None


async def get_manager() -> DCABotManager:
    """Get or create the unified manager (ONE manager for both paper + live).

    Creates a HybridExchangeClient that routes all orders to BOTH:
    - Paper client (simulated fills)
    - Live client (real Bybit orders)

    Both execute simultaneously — no async mirroring, atomic execution.
    """
    global _manager, _param_sync_task

    if _manager is None:
        # Verify live balance (will raise if < $1)
        balance = await fetch_real_balance()
        live_balance = balance.total_available_balance
        if live_balance < 1.0:
            raise ValueError(f"Insufficient balance for live: {live_balance:.8f} USDT")

        logger.warning(f"🔴 UNIFIED TRADING ACTIVE 🔴 Live balance: ${live_balance:.2f}")

        # Create both clients
        initial_balances = {"USDT": live_balance}
        paper_client = await create_exchange_client(
            TradingMode.PAPER, paper_initial_balances=initial_balances
        )
        if not isinstance(paper_client, PaperExchangeClient):
            raise TypeError("Paper client creation failed")

        live_client = await create_exchange_client(TradingMode.LIVE)

        # Hybrid client routes orders to BOTH
        hybrid_client = HybridExchangeClient(paper_client, live_client)

        # ONE manager using hybrid client
        _manager = DCABotManager(hybrid_client)
        await _manager.start()

        # Sync with best winner params
        await _sync_params_with_winner(_manager)

        # Auto-refresh params loop
        _param_sync_task = asyncio.create_task(_auto_refresh_params(_manager))

    return _manager


async def start_unified_trading() -> dict:
    """Start unified trading (both paper + live)."""
    global _trading_enabled
    _trading_enabled = True
    if _manager is None:
        try:
            await get_manager()
            return {"status": "started", "message": "Unified trading started (paper + live)"}
        except Exception as e:
            _trading_enabled = False
            return {"status": "error", "message": str(e)}
    return {"status": "already_running", "message": "Unified trading already running"}


async def stop_unified_trading() -> dict:
    """Stop unified trading."""
    global _trading_enabled
    _trading_enabled = False
    return {"status": "stopped", "message": "Unified trading stopped"}


async def close_all_positions() -> dict:
    """Close all open positions (live only, paper is simulation)."""
    global _manager
    if _manager is None:
        await get_manager()

    # Close live positions
    live_client = getattr(_manager.exchange, "live", None)
    if live_client is None:
        return {"status": "error", "message": "Live client not available"}

    try:
        position = live_client.position(SYMBOL)
        if position and position.qty != 0:
            side = "Sell" if position.qty > 0 else "Buy"
            logger.warning(f"CLOSING {side} {abs(position.qty)} {SYMBOL}")
            await live_client.place_market_order(
                SYMBOL, side, abs(position.qty), reduce_only=True
            )
            return {"status": "success", "message": f"Closed {abs(position.qty)} {SYMBOL}"}
        else:
            return {"status": "no_position", "message": "No open position"}
    except Exception as e:
        logger.error(f"Failed to close: {e}")
        return {"status": "error", "message": str(e)}


async def _sync_params_with_winner(manager: DCABotManager) -> None:
    """Sync manager bot with latest winner params."""
    result = get_best_current_winner()
    if not result:
        logger.info("No winner yet, using fallback params")
        config = {
            "side": "long",
            "leverage": DEFAULT_LEVERAGE,
            "dca_order_amount": 45.0,
            "first_order_amount": 20.0,
            "dca_max_order": 10,
            "dca_order_step_percent": 1.3,
            "dca_order_size_multiplier": 1.08,
            "dca_take_profit_percent": FIXED_TAKE_PROFIT_PERCENT,
            "funds_utilization_percent": DEFAULT_FUNDS_UTILIZATION_PERCENT,
        }
    else:
        config = row_params(result)
        config["dca_take_profit_percent"] = FIXED_TAKE_PROFIT_PERCENT

    bot = BotConfig(bot_name="unified-auto", pair=PAIR, **config)
    manager.add_bot(bot)
    logger.info(f"Synced unified bot with params: {config}")


async def _auto_refresh_params(manager: DCABotManager) -> None:
    """Periodically refresh params (every 8 hours)."""
    while True:
        try:
            await asyncio.sleep(8 * 3600)
            await _sync_params_with_winner(manager)
            logger.info("Refreshed unified params")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Param refresh failed: {e}")


async def shutdown_manager() -> None:
    """Clean shutdown."""
    global _manager, _param_sync_task
    if _param_sync_task:
        _param_sync_task.cancel()
        await asyncio.gather(_param_sync_task, return_exceptions=True)
    if _manager:
        await _manager.exchange.close()
        await _manager.stop()
        _manager = None
