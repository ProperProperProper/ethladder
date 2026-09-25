"""Paper (sandbox) exchange client — real Bybit LINEAR FUTURES market data
(ticker/kline/instrument precision) via an UNAUTHENTICATED public session,
but every order is simulated in-memory against a fake USDT margin
balance and a simulated position. No API key is ever needed to construct
this, and it can never place a real order: there is no code path here
that reaches an authenticated Bybit endpoint.

Position-based accounting (not spot balance debit/credit), tracking a
SIGNED qty (positive = long, negative = short): opening/adding in either
direction debits margin = notional/leverage from the USDT balance and
updates the position's VWAP average price; closing/reducing credits
back (released margin + realized PnL, which side profits from a price
move is determined by the position's own sign, not assumed long). This
mirrors real linear-futures margin mechanics and the "same margin, N x
bigger position" convention this port uses for leverage — see
strategy/backtest.py's build_ladder for the equivalent backtest-side math.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from symbot_python.exchange.base import (
    InstrumentPrecision,
    InsufficientMarginError,
    OrderResult,
    OrderSide,
    OrderStatus,
    Ticker,
    TradingMode,
)
from symbot_python.exchange.bybit_client import BybitSession
from symbot_python.exchange.risk_limits import get_maintenance_margin_rate
from symbot_python.exchange.ticker_stream import TickerStream
from symbot_python.exchange.utils import call_with_timeout, round_to_step, unwrap

CATEGORY = "linear"
MARGIN_ASSET = "USDT"
logger = logging.getLogger(__name__)


def _require_margin(available: float, required: float) -> None:
    if required > available:
        raise InsufficientMarginError(
            f"needs {required:.8f} margin but only {available:.8f} is available"
        )


@dataclass
class _FilledOrder:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float


@dataclass(frozen=True)
class PaperFill:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    opposite_price: float
    leverage: float


@dataclass
class PaperPosition:
    qty: float = 0.0  # SIGNED: positive = long, negative = short, 0 = flat
    avg_price: float = 0.0
    margin_committed: float = 0.0  # cumulative margin locked against this position

    @property
    def is_open(self) -> bool:
        return abs(self.qty) > 1e-12


class PaperExchangeClient:
    """PAPER trading client for linear perpetual futures. Constructed
    with only a public (no api_key/secret) Bybit session for market
    data, plus a starting USDT margin balance.
    """

    mode = TradingMode.PAPER

    def __init__(
        self,
        market_session: BybitSession,
        initial_balances: dict[str, float] | None = None,
        fee_rate_percent: float = 0.06,  # linear futures taker fee is lower than spot's
        ticker_stream: TickerStream | None = None,
    ):
        self._session = market_session
        self._precision_cache: dict[str, InstrumentPrecision] = {}
        self.margin_balance: float = (initial_balances or {}).get(MARGIN_ASSET, 10_000.0)
        self.fee_rate_percent = fee_rate_percent
        self._leverage: dict[str, float] = {}
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, _FilledOrder] = {}
        self.on_fill: Callable[[PaperFill], None] | None = None
        self.on_funding: Callable[[float], None] | None = None
        # Optional — every caller that omits it (existing tests included)
        # gets exactly the old always-REST behavior. When present,
        # get_ticker() reads from this instead of hitting REST every
        # tick; see ticker_stream.py's module docstring for why.
        self._ticker_stream = ticker_stream

    def position(self, symbol: str) -> PaperPosition:
        return self._positions.setdefault(symbol, PaperPosition())

    def close(self) -> None:
        """Stops the websocket feed, if one was given — soft-detected by
        callers via getattr rather than added to the ExchangeClient
        Protocol, since only paper mode owns a background connection to
        tear down; BybitClient/backtest have nothing to close."""
        if self._ticker_stream is not None:
            self._ticker_stream.stop()

    # -- account ----------------------------------------------------------------

    async def get_balance(self, coin: str | None = None) -> dict[str, float]:
        if coin and coin != MARGIN_ASSET:
            return {}
        return {MARGIN_ASSET: self.margin_balance}

    async def ensure_leverage(self, symbol: str, leverage: float) -> None:
        self._leverage[symbol] = leverage

    async def get_maintenance_margin_rate(self, symbol: str) -> float:
        return await call_with_timeout(asyncio.to_thread(get_maintenance_margin_rate, self._session, symbol, CATEGORY))

    async def get_current_funding_rate(self, symbol: str) -> float:
        """The rate that will settle at the symbol's next funding time —
        Bybit's get_tickers response already carries this on every linear
        perpetual (no separate history call needed for "what's the rate
        right now"). Used by dca_bot.py to simulate the same funding cost
        a real open position would actually incur, matching backtest.py's
        funding_events accounting so paper trading isn't systematically
        more profitable than a backtest of the identical strategy for a
        reason that has nothing to do with the strategy.
        """
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(self._session.get_tickers, category=CATEGORY, symbol=symbol))
        )
        items = result.get("list", [])
        if not items:
            return 0.0
        return float(items[0].get("fundingRate") or 0.0)

    def apply_funding_cost(self, cost_quote: float) -> None:
        """Nets accumulated funding into margin_balance ONCE, at deal
        close (called from dca_bot.py's _handle_sell with the deal's
        total accrued funding_cost_quote) — mirrors backtest.py's model
        exactly: funding is netted into profit_quote at the trade's
        close, not applied as a running mid-trade balance mutation.
        cost_quote > 0 is a cost (debited); < 0 is a credit.
        """
        self.margin_balance -= cost_quote
        if self.on_funding is not None:
            try:
                self.on_funding(cost_quote)
            except Exception:
                logger.exception("Paper funding observer failed")

    # -- market data (real data, from the public session) ------------------------

    async def get_precision(self, symbol: str, force_refresh: bool = False) -> InstrumentPrecision:
        if not force_refresh and symbol in self._precision_cache:
            return self._precision_cache[symbol]
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(
                self._session.get_instruments_info, category=CATEGORY, symbol=symbol
            ))
        )
        items = result.get("list", [])
        if not items:
            raise ValueError(f"Unknown Bybit {CATEGORY} symbol: {symbol}")
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

    async def get_ticker(self, symbol: str) -> Ticker:
        if self._ticker_stream is not None:
            self._ticker_stream.ensure_subscribed(symbol)
            cached = self._ticker_stream.get_price(symbol)
            if cached is not None:
                return cached
        result = unwrap(
            await call_with_timeout(asyncio.to_thread(self._session.get_tickers, category=CATEGORY, symbol=symbol))
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
                category=CATEGORY,
                symbol=symbol,
                interval=interval,
                limit=limit,
            ))
        )
        rows = result.get("list", [])
        candles = [
            [float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
            for r in rows
        ]
        candles.reverse()
        return candles

    # -- simulated orders (position-based, leverage-aware) -----------------------

    async def place_market_order(self, symbol: str, side: OrderSide, qty: float) -> OrderResult:
        """Buy/Sell here mean exactly what they mean on a real one-way-mode
        linear futures account: Buy opens/adds to a long OR closes/reduces
        an existing short; Sell opens/adds to a short OR closes/reduces an
        existing long — which one depends on the position's CURRENT sign,
        not on the order side alone. (A prior version of this method
        treated every "Sell" as closing a long and every "Buy" as opening
        one, unconditionally — a short's opening "Sell" against a flat
        position silently no-opped instead of opening anything, and its
        closing "Buy" was misread as opening a brand-new long. Every short
        paper deal was losing fees on phantom no-op fills and leaving a
        real, uncleared phantom long position behind for the next deal
        on the same symbol to inherit.)
        """
        ticker = await self.get_ticker(symbol)
        # Cross the spread like a real market order would: Buy fills at the
        # ask (the price you actually pay taking liquidity), Sell fills at
        # the bid. opposite_price is what the OPPOSITE side would have
        # filled at simultaneously — reverse_paper.py's mirror() uses it to
        # open the opposite-side position at a realistic price, not the
        # same price the original side got (that would silently give the
        # reverse account a zero-spread, unrealistically-cheap fill on
        # every mirrored trade). A prior version used the bid/ask midpoint
        # for both, which eliminated spread cost entirely from paper P&L —
        # made paper trading look more profitable than the equivalent live
        # trade ever could be.
        if ticker.bid and ticker.ask:
            if side == "Buy":
                fill_price, opposite_price = ticker.ask, ticker.bid
            else:
                fill_price, opposite_price = ticker.bid, ticker.ask
        else:
            fill_price = opposite_price = ticker.last
        return self.place_market_order_at_price(symbol, side, qty, fill_price, opposite_price)

    async def place_limit_order(self, symbol: str, side: OrderSide, qty: float, price: float) -> OrderResult:
        """Paper trading simulation of POST_ONLY limit order. Fills at limit price."""
        # In paper trading, we just fill at the specified limit price
        # Real Bybit would reject if crossing spread, but paper assumes liquidity
        opposite_price = None  # Not used for limit orders in paper
        return self.place_market_order_at_price(symbol, side, qty, price, opposite_price)

    def place_market_order_at_price(
        self, symbol: str, side: OrderSide, qty: float, fill_price: float,
        opposite_price: float | None = None, leverage: float | None = None,
    ) -> OrderResult:
        """Simulate a fill at a known public quote; never calls a trading endpoint."""
        if leverage is not None:
            self._leverage[symbol] = leverage
        leverage = self._leverage.get(symbol, 1.0)
        pos = self.position(symbol)
        notional = qty * fill_price
        fee = notional * (self.fee_rate_percent / 100)

        signed_qty = qty if side == "Buy" else -qty
        # Flat, or the order deepens the position already held (long+Buy
        # or short+Sell) -> opens/adds. Otherwise it's the reducing side.
        opens_or_adds = pos.qty == 0 or (pos.qty > 0) == (signed_qty > 0)

        if opens_or_adds:
            margin_required = notional / leverage
            _require_margin(self.margin_balance, margin_required + fee)
            new_qty = pos.qty + signed_qty
            pos.avg_price = (
                (pos.avg_price * abs(pos.qty) + fill_price * qty) / abs(new_qty) if new_qty else 0.0
            )
            pos.qty = new_qty
            pos.margin_committed += margin_required
            self.margin_balance -= margin_required + fee
        else:
            # Reduces (long closed by Sell, short closed by Buy). Long
            # profits as price rises above its average; short profits as
            # price falls below it.
            close_qty = min(qty, abs(pos.qty))
            pnl_per_unit = (fill_price - pos.avg_price) if pos.qty > 0 else (pos.avg_price - fill_price)
            raw_realized_pnl = close_qty * pnl_per_unit
            margin_released = pos.margin_committed * (close_qty / abs(pos.qty)) if pos.qty else 0.0
            # Isolated margin: the most this position can ever lose is
            # the margin actually committed to it — mirrors a real
            # exchange force-liquidating at (approximately) the point
            # losses would consume that margin, rather than reaching
            # into the rest of the account. Without this, a big enough
            # single-tick adverse move landing worse than the
            # liquidation-price check anticipated (the same "real fills
            # land at the actual tick price" gap already documented for
            # the OPEN side) could debit far more than this position
            # ever had backing it — verified: a 10x long dropping 25% in
            # one tick took balance from ~$4 to -$1500 under the old
            # uncapped math.
            realized_pnl = max(raw_realized_pnl, -margin_released)

            remaining_qty = qty - close_qty
            flip_margin = 0.0
            if remaining_qty > 1e-12:
                # The order's size exceeds the open position: the excess
                # flips straight into a new position in the ORDER's own
                # direction, same as a real exchange netting a market
                # order against an existing position. Checked BEFORE any
                # mutation (this close included, using the balance it
                # WOULD produce) so a flip that can't be margined rejects
                # the WHOLE order atomically — a caller catching
                # InsufficientMarginError must be able to trust that
                # nothing happened, not that the close silently went
                # through while only the flip failed.
                flip_margin = (remaining_qty * fill_price) / leverage
                _require_margin(self.margin_balance + margin_released + realized_pnl - fee, flip_margin)

            self.margin_balance += margin_released + realized_pnl - fee
            direction = 1 if pos.qty > 0 else -1
            pos.qty -= direction * close_qty
            pos.margin_committed -= margin_released
            if not pos.is_open:
                pos.qty = 0.0
                pos.avg_price = 0.0
                pos.margin_committed = 0.0

            if remaining_qty > 1e-12:
                flip_direction = 1 if side == "Buy" else -1
                pos.qty = flip_direction * remaining_qty
                pos.avg_price = fill_price
                pos.margin_committed = flip_margin
                self.margin_balance -= flip_margin

        order_id = f"paper-{uuid.uuid4().hex[:16]}"
        self._orders[order_id] = _FilledOrder(
            order_id=order_id, symbol=symbol, side=side, qty=qty, price=fill_price, fee=fee
        )
        result = OrderResult(order_id=order_id, symbol=symbol, side=side, qty_requested=qty)
        if self.on_fill is not None:
            try:
                self.on_fill(PaperFill(order_id, symbol, side, qty, fill_price,
                                       opposite_price or fill_price, leverage))
            except Exception:
                logger.exception("Paper fill observer failed for %s", order_id)
        return result

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        order = self._orders.get(order_id)
        if order is None:
            return OrderStatus(
                order_id=order_id, status="unknown", avg_price=0, cum_exec_qty=0,
                cum_exec_value=0, cum_exec_fee=0,
            )
        return OrderStatus(
            order_id=order_id,
            status="filled",
            avg_price=order.price,
            cum_exec_qty=order.qty,
            cum_exec_value=order.qty * order.price,
            cum_exec_fee=order.fee,
        )

    async def verify_order(self, symbol: str, order_id: str) -> OrderStatus:
        # Paper fills are always instant and always confirmed — no polling
        # needed since there's no real exchange latency to wait out.
        return await self.get_order_status(symbol, order_id)
