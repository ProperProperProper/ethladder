"""Shared types + the ExchangeClient protocol that both the live Bybit
client and the paper (sandbox) client implement identically, so the
strategy engine never has to know which one it's talking to.

Trading mode is a first-class, explicit concept: a bot never trades live
by accident. `TradingMode.LIVE` is the only mode that can place a real
order, and building a live client requires a separate explicit
confirmation step (see exchange/factory.py) — never just a config default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

OrderSide = Literal["Buy", "Sell"]


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class InstrumentPrecision:
    symbol: str
    tick_size: float
    qty_step: float
    min_order_qty: float
    min_order_amt: float


@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    volume_24h_base: float
    turnover_24h_quote: float


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    qty_requested: float


@dataclass
class OrderStatus:
    order_id: str
    status: str  # "open" | "filled" | "cancelled" | "rejected" | "unknown"
    avg_price: float
    cum_exec_qty: float
    cum_exec_value: float
    cum_exec_fee: float


class InsufficientMarginError(Exception):
    """An exchange client should raise this from place_market_order
    rather than ever letting an order fill past the account's actual
    available margin — real exchanges reject an unmargin-able order (or
    force a liquidation before equity could go negative); a client must
    never silently debit its balance below what it actually has.
    dca_bot.py catches this the same way as a timeout: pause the deal
    for visibility rather than treat it as a fatal engine crash.
    """


@runtime_checkable
class ExchangeClient(Protocol):
    """The exact surface strategy/dca_bot.py is written against. Both
    BybitClient (live) and PaperExchangeClient (backtest/paper) implement
    this so the engine code is identical regardless of trading mode.

    Leverage lives at the (client, symbol) level via ensure_leverage —
    matching Bybit's own model where leverage is a property of the
    symbol's position, not a per-order parameter — rather than being
    passed into place_market_order.
    """

    mode: TradingMode

    async def get_precision(self, symbol: str, force_refresh: bool = False) -> InstrumentPrecision: ...
    def filter_price(self, precision: InstrumentPrecision, price: float) -> float: ...
    def filter_amount(self, precision: InstrumentPrecision, qty: float) -> float: ...
    async def get_ticker(self, symbol: str) -> Ticker: ...
    async def get_kline(self, symbol: str, interval: str, limit: int = 200) -> list[list[float]]: ...
    async def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> OrderResult: ...
    async def place_limit_order(self, symbol: str, side: OrderSide, qty: float, price: float) -> OrderResult: ...
    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus: ...
    async def verify_order(self, symbol: str, order_id: str) -> OrderStatus: ...
    async def get_balance(self, coin: str | None = None) -> dict[str, float]: ...
    async def ensure_leverage(self, symbol: str, leverage: float) -> None: ...
    async def get_maintenance_margin_rate(self, symbol: str) -> float: ...
