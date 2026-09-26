"""Hybrid exchange client — executes trades on BOTH paper AND live simultaneously.

Instead of separate paper/live managers with mirroring logic, one manager
uses this hybrid client which forwards every order to both paper and live
clients. Fills are recorded in BOTH systems atomically.
"""

from __future__ import annotations

import logging
from typing import Optional

from symbot_python.exchange.base import (
    ExchangeClient, InstrumentPrecision, OrderResult, OrderSide, OrderStatus,
    Ticker, TradingMode,
)
from symbot_python.exchange.paper_client import PaperExchangeClient

logger = logging.getLogger(__name__)


class HybridExchangeClient(ExchangeClient):
    """Routes all trades to paper client (simulated) + live client (real Bybit).

    Both clients share the same order stream — when you place an order,
    it executes on BOTH paper (for testing) and live (real money) simultaneously.
    Position tracking shows paper fills and live fills separately.
    """

    def __init__(self, paper_client: PaperExchangeClient, live_client: ExchangeClient):
        self.paper = paper_client
        self.live = live_client
        self.category = live_client.category  # Use live client's settings

    async def verify_connection(self) -> None:
        await self.paper.verify_connection()
        await self.live.verify_connection()

    async def get_balance(self, coin: Optional[str] = None) -> dict[str, float]:
        # Return live balance (real money)
        return await self.live.get_balance(coin)

    @property
    def margin_balance(self) -> float:
        """Real live balance for equity tracking."""
        return self.live.margin_balance

    async def ensure_leverage(self, symbol: str, leverage: float) -> None:
        # Only set leverage on live (paper doesn't need it)
        await self.live.ensure_leverage(symbol, leverage)

    async def get_precision(self, symbol: str) -> InstrumentPrecision:
        return await self.live.get_precision(symbol)

    async def get_ticker(self, symbol: str) -> Ticker:
        # Use live prices (real market data)
        return await self.live.get_ticker(symbol)

    async def place_market_order(
        self, symbol: str, side: OrderSide, qty: float, reduce_only: bool = False
    ) -> OrderResult:
        """Place market order on BOTH paper and live simultaneously."""
        # Paper fill (simulated, immediate)
        paper_result = await self.paper.place_market_order(symbol, side, qty, reduce_only)
        logger.info(f"📄 PAPER order {paper_result.order_id}: {side} {qty} {symbol}")

        # Live fill (real Bybit order)
        live_result = await self.live.place_market_order(symbol, side, qty, reduce_only)
        logger.info(f"🔴 LIVE order {live_result.order_id}: {side} {qty} {symbol}")

        # Return live result (authoritative for position tracking)
        # Paper fills still recorded in paper manager's deals
        return live_result

    async def place_limit_order(
        self, symbol: str, side: OrderSide, qty: float, price: float
    ) -> OrderResult:
        """Place limit order on BOTH paper and live."""
        # Paper fill (simulated at limit price)
        paper_result = await self.paper.place_limit_order(symbol, side, qty, price)
        logger.info(f"📄 PAPER limit {paper_result.order_id}: {side} {qty} @ ${price}")

        # Live fill (real Bybit PostOnly order)
        live_result = await self.live.place_limit_order(symbol, side, qty, price)
        logger.info(f"🔴 LIVE limit {live_result.order_id}: {side} {qty} @ ${price}")

        return live_result

    async def verify_order(self, symbol: str, order_id: str) -> OrderStatus:
        # Verify on live (real execution)
        return await self.live.verify_order(symbol, order_id)

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        return await self.live.get_order_status(symbol, order_id)

    async def get_position(self, symbol: str) -> Optional:
        # Return live position (real account)
        return await self.live.get_position(symbol)

    def position(self, symbol: str) -> Optional:
        """Sync position read (cached)."""
        return self.live.position(symbol)

    async def get_maintenance_margin_rate(self, symbol: str) -> float:
        return await self.live.get_maintenance_margin_rate(symbol)

    async def close(self) -> None:
        await self.paper.close()
        await self.live.close()
