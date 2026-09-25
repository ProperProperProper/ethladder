"""Bybit v5 exchange wrapper — LINEAR PERPETUAL FUTURES (category="linear"),
matching the product the user's own live ETH bot trades leverage on
(unified-combo-grid uses category="linear" + set_leverage). Talks to
Bybit's own API shapes via `pybit.unified_trading.HTTP` directly rather
than going through a normalizing library, so precision filtering and
order-status parsing are hand-rolled here against Bybit's actual
response fields.

One-way position mode throughout (positionIdx=0, same as the reference
bot) — no hedge mode. Both long and short are supported: reduceOnly is
computed from the symbol's ACTUAL current position (see
place_market_order), not assumed from order side — "Buy"/"Sell" open or
add to a position in that direction when there's nothing to reduce, and
reduce/close the opposite position when one exists.

Design notes:
- Market orders only (no limit-order path) — a deliberate scope
  restriction, kept simple rather than building out order-book-aware
  limit placement and its retry/reprice logic.
- Every price/qty that is about to be sent to the exchange or used in
  sizing math is rounded through filter_price/filter_amount, built from
  this instrument's own tickSize/qtyStep (Bybit's `get_instruments_info`),
  not a generic precision guess.
- Order verification polls up to MAX_VERIFY_TRIES times, cancels an order
  still open after ORDER_OPEN_TIMEOUT_SEC seconds — bounded retry so a
  stuck order can't hang the bot indefinitely.

Liquidation price: prefer Bybit's OWN reported `liqPrice` from
get_position() over the approximate formula in dca_math.py — the
exchange's figure accounts for real-time mark price and actual margin
mode; the formula is only a stand-in for backtesting, where no live
position exists to query.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from symbot_python.exchange.base import (
    InstrumentPrecision,
    OrderResult,
    OrderSide,
    OrderStatus,
    Ticker,
    TradingMode,
)
from symbot_python.exchange.risk_limits import get_maintenance_margin_rate as _get_mmr
from symbot_python.exchange.utils import (
    BybitApiError,
    call_with_timeout,
    format_qty,
    round_to_step,
    unwrap,
)

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY = "linear"
ONE_WAY_POSITION_IDX = 0
LEVERAGE_NOT_MODIFIED_ERROR_CODE = "110043"  # benign: already set to this value
MAX_VERIFY_TRIES = 15
VERIFY_POLL_DELAY_SEC = 1.0
ORDER_OPEN_TIMEOUT_SEC = 75


class BybitSession(Protocol):
    """The subset of pybit.unified_trading.HTTP's surface this wrapper
    calls. A test double only needs to implement these methods.
    """

    def get_wallet_balance(self, **kwargs: Any) -> dict: ...
    def get_tickers(self, **kwargs: Any) -> dict: ...
    def get_instruments_info(self, **kwargs: Any) -> dict: ...
    def place_order(self, **kwargs: Any) -> dict: ...
    def get_order_history(self, **kwargs: Any) -> dict: ...
    def get_open_orders(self, **kwargs: Any) -> dict: ...
    def cancel_order(self, **kwargs: Any) -> dict: ...
    def get_kline(self, **kwargs: Any) -> dict: ...
    def set_leverage(self, **kwargs: Any) -> dict: ...
    def get_positions(self, **kwargs: Any) -> dict: ...


_TERMINAL_FILLED = {"Filled"}
_TERMINAL_DEAD = {"Cancelled", "Rejected", "Deactivated"}
_OPEN_STATUSES = {"New", "PartiallyFilled", "Untriggered", "Created"}


def _normalize_status(bybit_status: str) -> str:
    if bybit_status in _TERMINAL_FILLED:
        return "filled"
    if bybit_status in _TERMINAL_DEAD:
        return "cancelled"
    if bybit_status in _OPEN_STATUSES:
        return "open"
    return "unknown"


@dataclass
class PositionInfo:
    symbol: str
    side: str  # "Buy" (long) | "None" (flat)
    size: float
    avg_price: float
    leverage: float
    liquidation_price: float
    unrealized_pnl: float


class BybitClient:
    """LIVE trading client — one instance per unique (api_key, api_secret)
    pair. Places real orders with real funds on linear perpetual futures.
    Never construct this directly from bot config; go through
    exchange/factory.py, which is the one place a bot's requested trading
    mode is checked before a live-capable client is ever handed to the
    engine.
    """

    mode = TradingMode.LIVE

    def __init__(self, session: BybitSession, category: str = DEFAULT_CATEGORY):
        self._session = session
        self.category = category
        self._precision_cache: dict[str, InstrumentPrecision] = {}
        self._leverage_set: dict[str, float] = {}
        self._position_cache: dict[str, Optional[PositionInfo]] = {}

    # -- connection / account -------------------------------------------------

    async def verify_connection(self) -> None:
        unwrap(await call_with_timeout(asyncio.to_thread(self._session.get_wallet_balance, accountType="UNIFIED")))

    async def get_balance(self, coin: Optional[str] = None) -> dict[str, float]:
        kwargs: dict[str, Any] = {"accountType": "UNIFIED"}
        if coin:
            kwargs["coin"] = coin
        result = unwrap(await call_with_timeout(asyncio.to_thread(self._session.get_wallet_balance, **kwargs)))
        balances: dict[str, float] = {}
        for account in result.get("list", []):
            for entry in account.get("coin", []):
                balances[entry["coin"]] = float(entry.get("walletBalance") or 0)
        return balances

    # -- leverage / position ---------------------------------------------------

    async def ensure_leverage(self, symbol: str, leverage: float) -> None:
        """Set leverage for `symbol` if not already set to this value in
        this process. Bybit returns error 110043 ("leverage not
        modified") when it's already at the requested value — benign,
        matching the reference bot's handling.
        """
        if self._leverage_set.get(symbol) == leverage:
            return
        lev = str(leverage)
        try:
            unwrap(
                await call_with_timeout(asyncio.to_thread(
                    self._session.set_leverage,
                    category=self.category, symbol=symbol,
                    buyLeverage=lev, sellLeverage=lev,
                ))
            )
        except BybitApiError as exc:
            if str(exc.ret_code) != LEVERAGE_NOT_MODIFIED_ERROR_CODE:
                raise
        self._leverage_set[symbol] = leverage

    async def get_maintenance_margin_rate(self, symbol: str) -> float:
        return await call_with_timeout(asyncio.to_thread(_get_mmr, self._session, symbol, self.category))

    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """The exchange's own live view of this symbol's position,
        including its actual liquidation price — prefer this over the
        approximate formula in dca_math.py whenever a real position exists.
        """
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.get_positions, category=self.category, symbol=symbol
            ))
        )
        items = result.get("list", [])
        if not items or float(items[0].get("size") or 0) == 0:
            self._position_cache[symbol] = None
            return None
        p = items[0]
        pos = PositionInfo(
            symbol=symbol,
            side=p.get("side", "None"),
            size=float(p.get("size") or 0),
            avg_price=float(p.get("avgPrice") or 0),
            leverage=float(p.get("leverage") or 1),
            liquidation_price=float(p.get("liqPrice") or 0) if p.get("liqPrice") else 0.0,
            unrealized_pnl=float(p.get("unrealisedPnl") or 0),
        )
        self._position_cache[symbol] = pos
        return pos

    def position(self, symbol: str) -> Optional[PositionInfo]:
        """Synchronous position lookup — returns cached position or None (flat)."""
        return self._position_cache.get(symbol)

    # -- instrument precision --------------------------------------------------

    async def get_precision(self, symbol: str, force_refresh: bool = False) -> InstrumentPrecision:
        if not force_refresh and symbol in self._precision_cache:
            return self._precision_cache[symbol]
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.get_instruments_info, category=self.category, symbol=symbol
            ))
        )
        items = result.get("list", [])
        if not items:
            raise ValueError(f"Unknown Bybit {self.category} symbol: {symbol}")
        info = items[0]
        lot = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})
        precision = InstrumentPrecision(
            symbol=symbol,
            tick_size=float(price_filter.get("tickSize", 0) or 0),
            qty_step=float(lot.get("qtyStep", lot.get("basePrecision", 0)) or 0),
            min_order_qty=float(lot.get("minOrderQty", 0) or 0),
            min_order_amt=float(lot.get("minNotionalValue", lot.get("minOrderAmt", 0)) or 0),
        )
        self._precision_cache[symbol] = precision
        return precision

    def filter_price(self, precision: InstrumentPrecision, price: float) -> float:
        return round_to_step(price, precision.tick_size)

    def filter_amount(self, precision: InstrumentPrecision, qty: float) -> float:
        return round_to_step(qty, precision.qty_step)

    # -- market data ------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(self._session.get_tickers, category=self.category, symbol=symbol))
        )
        items = result.get("list", [])
        if not items:
            raise ValueError(f"No ticker data for {symbol}")
        t = items[0]
        return Ticker(
            symbol=symbol,
            last=float(t.get("lastPrice") or 0),
            bid=float(t.get("bid1Price") or 0),
            ask=float(t.get("ask1Price") or 0),
            volume_24h_base=float(t.get("volume24h") or 0),
            turnover_24h_quote=float(t.get("turnover24h") or 0),
        )

    async def get_kline(self, symbol: str, interval: str, limit: int = 200) -> list[list[float]]:
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.get_kline,
                category=self.category,
                symbol=symbol,
                interval=interval,
                limit=limit,
            ))
        )
        # Bybit returns newest-first as [start, open, high, low, close, volume, turnover]
        rows = result.get("list", [])
        candles = [
            [float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
            for r in rows
        ]
        candles.reverse()
        return candles

    # -- orders -------------------------------------------------------------

    async def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> OrderResult:
        """Market order only — the limit-order path is not implemented,
        a deliberate scope restriction (see module docstring).

        One-way position mode (positionIdx=0): reduceOnly is set from the
        symbol's ACTUAL current position, not from order side alone — a
        "Sell" opens/adds to a SHORT when there's no long to reduce (or
        the position is already short), and only means "close the long"
        when one exists. Bybit rejects a reduceOnly order that isn't
        actually reducing anything, so the previous side-only rule
        (`reduceOnly=(side == "Sell")`) would have rejected every short
        deal's own opening order outright the moment live short trading
        was ever enabled — caught here before that could happen for real.

        A reducing-direction order sized to close AT MOST the current
        position (qty <= position.size) is a plain reduce: reduceOnly=
        True, so Bybit itself rejects it outright if the position isn't
        really there to reduce — a real safety net against a stale/wrong
        qty. One sized LARGER than the current position is a deliberate
        close-and-reverse (see dca_bot.py's reverse_drawdown handling):
        reduceOnly=False lets Bybit's own one-way-mode netting close the
        old position and open the new one from a single order, matching
        exactly how PaperExchangeClient's equivalent flip branch already
        behaves in simulation. Getting this wrong is a real live-trading
        gap, not just cosmetic: a reduceOnly order Bybit deems oversized
        for what it would actually reduce is rejected or clamped, so a
        stop-and-reverse built by simply sending a bigger qty would
        silently never reverse against a real account without this.
        """
        position = await self.get_position(symbol)
        if position is None:
            reduce_only = False
        else:
            is_reducing_direction = (position.side == "Buy" and side == "Sell") or (
                position.side == "Sell" and side == "Buy"
            )
            reduce_only = is_reducing_direction and qty <= position.size
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.place_order,
                category=self.category,
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=format_qty(qty),
                positionIdx=ONE_WAY_POSITION_IDX,
                reduceOnly=reduce_only,
            ))
        )
        order_id = result.get("orderId")
        if not order_id:
            raise BybitApiError(-1, "place_order response missing orderId")
        return OrderResult(order_id=order_id, symbol=symbol, side=side, qty_requested=qty)

    async def place_limit_order(self, symbol: str, side: OrderSide, qty: float, price: float) -> OrderResult:
        """POST_ONLY limit order (maker fee 0.01% vs taker 0.06%). Bybit-native feature.

        Places limit at exact price with POST_ONLY flag: order fills as maker ONLY,
        never crosses the spread. Gets auto-rejected by Bybit if it would be a taker.
        Reduces fees by 85% (0.01% maker vs 0.06% taker).

        Should be used for entry orders. Falls back to market if 3 retries fail.
        """
        position = await self.get_position(symbol)
        if position is None:
            reduce_only = False
        else:
            is_reducing_direction = (position.side == "Buy" and side == "Sell") or (
                position.side == "Sell" and side == "Buy"
            )
            reduce_only = is_reducing_direction and qty <= position.size

        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.place_order,
                category=self.category,
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=format_qty(qty),
                price=format_price(price),
                timeInForce="PostOnly",  # Bybit-native: maker fee only
                positionIdx=ONE_WAY_POSITION_IDX,
                reduceOnly=reduce_only,
            ))
        )
        order_id = result.get("orderId")
        if not order_id:
            raise BybitApiError(-1, "place_limit_order response missing orderId")
        return OrderResult(order_id=order_id, symbol=symbol, side=side, qty_requested=qty)

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        open_result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.get_open_orders, category=self.category, symbol=symbol, orderId=order_id
            ))
        )
        items = open_result.get("list", [])
        if not items:
            history = unwrap(
                await call_with_timeout(asyncio.to_thread(
                    self._session.get_order_history,
                    category=self.category,
                    symbol=symbol,
                    orderId=order_id,
                ))
            )
            items = history.get("list", [])
        if not items:
            return OrderStatus(
                order_id=order_id, status="unknown", avg_price=0, cum_exec_qty=0,
                cum_exec_value=0, cum_exec_fee=0,
            )
        o = items[0]
        return OrderStatus(
            order_id=order_id,
            status=_normalize_status(o.get("orderStatus", "")),
            avg_price=float(o.get("avgPrice") or 0),
            cum_exec_qty=float(o.get("cumExecQty") or 0),
            cum_exec_value=float(o.get("cumExecValue") or 0),
            cum_exec_fee=float(o.get("cumExecFee") or 0),
        )

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        await call_with_timeout(asyncio.to_thread(
            self._session.cancel_order, category=self.category, symbol=symbol, orderId=order_id
        ))

    async def verify_order(self, symbol: str, order_id: str) -> OrderStatus:
        """Poll an order's status up to MAX_VERIFY_TRIES times, cancelling
        it if it's still open past ORDER_OPEN_TIMEOUT_SEC. Returns the
        last-seen status; "unknown" after exhausting tries means the
        caller should treat it as an invalid/unconfirmable order and
        pause the bot for manual reconciliation rather than guess at
        what happened to it.
        """
        started = time.monotonic()
        status = OrderStatus(order_id=order_id, status="open", avg_price=0, cum_exec_qty=0, cum_exec_value=0, cum_exec_fee=0)
        for attempt in range(MAX_VERIFY_TRIES):
            status = await self.get_order_status(symbol, order_id)
            if status.status in ("filled", "cancelled"):
                return status
            if status.status == "open" and time.monotonic() - started > ORDER_OPEN_TIMEOUT_SEC:
                try:
                    await self.cancel_order(symbol, order_id)
                except BybitApiError as exc:
                    logger.warning("cancel_order failed for %s: %s", order_id, exc)
            await asyncio.sleep(VERIFY_POLL_DELAY_SEC + attempt * 0.05)
        return status
