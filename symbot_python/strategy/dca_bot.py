"""Per-deal async engine. One DCABotEngine instance runs ONE deal, as one
asyncio.Task with its own tick loop — an independent loop per deal, with
no shared scheduler tick, so one deal's timing never blocks another's.

Known simplification (documented, not hidden): order placement uses a
single place-then-verify call rather than partial-fill-credit /
long-horizon invalid-order-retry machinery. A fill that can't be
verified pauses the deal for manual reconcile instead of attempting
automatic recovery. This is a real gap to close before trusting this
with meaningful size, but every guard module (stop-loss, price sanity)
and the trigger-precedence rules are implemented in full.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime
from typing import Awaitable, Callable, Optional, Protocol

from symbot_python.exchange.base import ExchangeClient, InstrumentPrecision, InsufficientMarginError
from symbot_python.strategy import dca_math, price_guard
from symbot_python.strategy.backtest import MIN_WIN_PRICE_PCT, WIN_FEE_MULT
from symbot_python.strategy.models import BotConfig, Deal, DealStatus, to_exchange_symbol
from symbot_python.strategy.stop_loss import StopLossInput, StopLossResult, evaluate as evaluate_stop_loss
from symbot_python.exchange.dip_analysis_service import get_dip_analysis_service
from symbot_python.exchange.dip_calibration_engine import get_calibration_engine
from symbot_python.ml.trade_memory import append_live_paper_trade

logger = logging.getLogger(__name__)

TICK_INTERVAL_SEC = 2.0
RETRY_INTERVAL_SEC = 1.0
# No exchange call in the tick loop had a timeout before this — a single
# stalled live network call (get_ticker/place_market_order/verify_order)
# could freeze a deal's loop forever, with no exception and no log line,
# since the loop just never gets back around to checking cancel/panic
# flags at the top of the next _tick(). Observed directly: a real paper
# deal stopped responding to Cancel/Panic sell with nothing in the logs
# to explain why. A bounded timeout guarantees the loop always comes back
# around within a few seconds, cancel/panic included, instead of hanging
# indefinitely on one bad network call.
EXCHANGE_CALL_TIMEOUT_SEC = 10.0
# Bybit perpetuals settle funding every 8h, at fixed UTC boundaries
# (00:00/08:00/16:00) — i.e. every multiple of this many seconds since
# the epoch, since 1970-01-01T00:00:00Z is itself one such boundary.
FUNDING_INTERVAL_SEC = 8 * 3600


class CircuitBreaker(Protocol):
    @property
    def active(self) -> bool: ...


class _Tick:
    RETRY = "retry"
    DONE = "done"


OnComplete = Callable[[Deal], Awaitable[None]]


class DCABotEngine:
    def __init__(
        self,
        bot: BotConfig,
        deal: Deal,
        exchange: ExchangeClient,
        on_complete: Optional[OnComplete] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        maintenance_margin_rate: float = 0.0,
    ):
        self.bot = replace(bot)
        deal.config = self.bot
        self.deal = deal
        self.exchange = exchange
        self.on_complete = on_complete
        self.circuit_breaker = circuit_breaker
        self.maintenance_margin_rate = maintenance_margin_rate
        self._stop_requested = False
        self._cancel_requested = False
        self._panic_requested = False
        self._last_stop_loss_result: Optional[StopLossResult] = None

        # DIP analysis (drawdown profiting)
        self.dip_service = None
        self.calibration = None
        self.current_dip_record = None

    # -- external controls: stop/cancel/panic actions requested from outside the tick loop --

    def request_stop(self) -> None:
        """Detach monitoring only — no market action, deal stays open in the DB."""
        self._stop_requested = True

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def request_panic_sell(self) -> None:
        self._panic_requested = True

    def pause(self, pause: bool = True, pause_buy: bool = False, pause_sell: bool = False, reason: Optional[str] = None) -> None:
        self.deal.paused = pause
        self.deal.paused_buy = pause_buy
        self.deal.paused_sell = pause_sell
        self.deal.pause_reason = reason

    async def _call(self, awaitable):
        """Bounds every exchange call to EXCHANGE_CALL_TIMEOUT_SEC. Raises
        asyncio.TimeoutError on a stall — callers catch that AND OSError
        (covers ConnectionResetError/BrokenPipeError/ssl errors, plus
        requests.exceptions.RequestException, which subclasses OSError)
        and retry next tick (see _tick) rather than let one bad or
        interrupted network call either freeze this deal's loop forever,
        or — before this was broadened past just TimeoutError — get
        treated as a fatal engine crash by dca_bot_manager's
        _run_engine_and_recover and force-close a perfectly healthy deal
        over a routine dropped TCP connection. A real API-level error
        (pybit's InvalidRequestError/FailedRequestError) is NOT an
        OSError and still propagates as fatal, which is correct — that's
        a real bug or bad config, not network flakiness.
        """
        return await asyncio.wait_for(awaitable, timeout=EXCHANGE_CALL_TIMEOUT_SEC)

    async def _accrue_funding_if_due(self, price: float) -> None:
        """Charges/credits real funding for every 8h boundary crossed
        since the deal opened (or since the last check) — mirrors
        backtest.py's funding_events accounting so paper/live isn't
        systematically more profitable than a backtest of the identical
        strategy purely because paper trading never modeled this real
        cost. Accumulates into deal.funding_cost_quote; netted into the
        realized result once, at close (_handle_sell) — never mutates
        balance mid-trade, matching backtest.py's single net-at-close model.

        Soft feature-detection: not every ExchangeClient implements
        get_current_funding_rate/apply_funding_cost (only PaperExchangeClient
        does as of this writing — BybitClient's live equivalent is a
        follow-up, and live trading is unreachable without the separate
        confirm_live gate regardless), so this silently no-ops rather than
        raising when the exchange doesn't support it.
        """
        get_rate = getattr(self.exchange, "get_current_funding_rate", None)
        if get_rate is None:
            return
        now = time.time()
        prev = self.deal.last_funding_settlement_ts or now
        crossings = int(now // FUNDING_INTERVAL_SEC) - int(prev // FUNDING_INTERVAL_SEC)
        if crossings <= 0:
            self.deal.last_funding_settlement_ts = prev
            return
        self.deal.last_funding_settlement_ts = now
        try:
            rate = await self._call(get_rate(to_exchange_symbol(self.bot.pair)))
        except (asyncio.TimeoutError, OSError):
            logger.warning("get_current_funding_rate timed out or hit a network error — skipping this funding application.")
            return
        last_filled = self.deal.orders[self.deal.filled_count - 1]
        notional = last_filled.qty_sum * price
        is_short = self.bot.side == "short"
        # Long PAYS when rate>0; short is the exact mirror — same sign
        # convention as backtest.py's funding_cost_accum line.
        self.deal.funding_cost_quote += notional * rate * crossings * (-1.0 if is_short else 1.0)

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        # Initialize DIP analysis services
        self.dip_service = await get_dip_analysis_service()
        self.calibration = await get_calibration_engine()

        symbol = to_exchange_symbol(self.bot.pair)
        precision = await self.exchange.get_precision(symbol)
        while True:
            if self._stop_requested:
                return
            outcome, delay = await self._tick(symbol, precision)
            if outcome == _Tick.DONE:
                if self.on_complete:
                    await self.on_complete(self.deal)
                return
            await asyncio.sleep(delay)

    async def _tick(self, symbol: str, precision: InstrumentPrecision) -> tuple[str, float]:
        # Inherent backtest-vs-live optimism gap, not a bug: backtest.py
        # evaluates liquidation/stop-loss/take-profit against a whole
        # CANDLE's high/low (the best/worst price theoretically reachable
        # within that bar), while this tick loop can only ever act on
        # actual discrete prices it happens to observe, TICK_INTERVAL_SEC
        # apart. A backtest can therefore report a take-profit fill (or a
        # liquidation) that a real/paper deal would miss if the real move
        # happened between two ticks — backtest results are a ceiling on
        # what live/paper can achieve, not a tick-for-tick guarantee of
        # identical fills. The trigger conditions and priority order
        # (liquidation > stop-loss/trailing > safety-order > take-profit)
        # are otherwise kept identical to backtest.py's tick loop.
        if self.circuit_breaker is not None and self.circuit_breaker.active:
            return _Tick.RETRY, RETRY_INTERVAL_SEC

        is_short = self.bot.side == "short"

        try:
            ticker = await self._call(self.exchange.get_ticker(symbol))
        except (asyncio.TimeoutError, OSError):
            logger.warning("get_ticker timed out or hit a network error for %s — retrying next tick.", symbol)
            return _Tick.RETRY, RETRY_INTERVAL_SEC
        price = ticker.last

        if self.deal.is_start == 0:
            if self._cancel_requested or self._panic_requested:
                # No position was ever opened — nothing to sell, no P&L
                # to realize, so this closes directly rather than
                # through _handle_sell (which assumes at least one
                # filled rung exists and would index into an empty
                # deal.orders fill). Checked BEFORE the paused check
                # below: a stuck-at-open deal must still be cancellable,
                # exactly like an in-flight one.
                reason = "panic_sell" if self._panic_requested else "cancel"
                self.deal.status = DealStatus.CLOSED
                self.deal.date_closed = time.time()
                self.deal.canceled = reason == "cancel"
                self.deal.panic_sell = reason == "panic_sell"
                self.deal.sell_data = {
                    "date": self.deal.date_closed, "price": None, "average": None,
                    "profit_percent": 0.0, "profit_quote": 0.0, "reason": reason,
                }
                return _Tick.DONE, 0.0
            if self.deal.paused:
                # Without this, a paused base order (verify failure or
                # insufficient margin) retried every single tick forever
                # — is_start stays 0 until the base order actually
                # fills, so nothing else in this method ever runs to
                # gate it, unlike the safety-order/take-profit paths
                # below which already check `not self.deal.paused`.
                # Verified live: an unguarded insufficient-margin
                # rejection grew from 427 to 850 place_market_order
                # calls within one second, hammering the exchange and
                # spamming the log at RETRY_INTERVAL_SEC cadence.
                # Consistent with this module's own documented design —
                # a pause here means manual reconcile, not automatic
                # recovery (see the module docstring) — but Cancel/Panic
                # (checked above) still get through regardless.
                return _Tick.RETRY, RETRY_INTERVAL_SEC
            return await self._handle_base_order(symbol, price)

        # Real funding accrues on a real position regardless of whether
        # this tick's price passes the sanity guard below — check it
        # first, using whatever price this tick actually got.
        await self._accrue_funding_if_due(price)

        last_filled = self.deal.orders[self.deal.filled_count - 1]
        sanity = price_guard.evaluate_price_sanity(price, last_filled.average)
        cancel_only = (not sanity.plausible) or price <= 0
        if cancel_only:
            # Never buy, sell, or evaluate a stop against a price we don't
            # trust — hold and re-check next tick (fail toward inaction).
            return _Tick.RETRY, RETRY_INTERVAL_SEC

        if self.bot.leverage > 1:
            liq_price = dca_math.calculate_liquidation_price(
                last_filled.average, self.bot.leverage, self.maintenance_margin_rate, side=self.bot.side,
            )
            # LONG is liquidated on the way DOWN; SHORT is liquidated on
            # the way UP — same asymmetry as backtest.py's tick loop.
            liquidated = price >= liq_price if is_short else price <= liq_price
            if liquidated:
                # Takes priority over everything else: on a real exchange
                # this would already have been force-closed. Live trading
                # may find the position already gone (closed externally)
                # by the time this sell lands — _handle_sell's own
                # verify_order/pause-for-reconcile path is what surfaces
                # that, same as any other unconfirmable fill.
                return await self._handle_sell(symbol, liq_price, reason="liquidated")

        is_stop_loss = False
        if self.bot.dca_stop_loss_enabled or self.bot.dca_trailing_stop_enabled:
            is_stop_loss = self._evaluate_stop_loss(price)

        # Checked only when stop-loss isn't already firing this tick — a
        # plain hard stop-loss (if configured) is the more fundamental
        # protection and always wins over an opportunistic reversal.
        is_reverse_drawdown = False
        if not is_stop_loss and self.bot.reverse_drawdown_percent:
            is_reverse_drawdown = self._evaluate_reverse_drawdown(price)

        if self._panic_requested:
            return await self._handle_sell(symbol, price, reason="panic_sell")
        if self._cancel_requested:
            return await self._handle_sell(symbol, price, reason="cancel")
        if is_stop_loss:
            return await self._handle_sell(symbol, price, reason="stop_loss")
        if is_reverse_drawdown:
            return await self._handle_sell(symbol, price, reason="reverse_drawdown")

        if not self.deal.paused and not self.deal.paused_buy:
            filled_new = await self._handle_safety_order(symbol, price)
            if filled_new:
                return _Tick.RETRY, TICK_INTERVAL_SEC

        # Monitor for dips and analyze bounce probability
        if not self.deal.paused and self.dip_service:
            dip_decision = await self._analyze_dip(symbol, price, ticker)
            if dip_decision and dip_decision.should_add_safety:
                if dip_decision.liquidation_risk != "HIGH":
                    # Place safety order from DIP analysis
                    await self._place_dip_safety_order(symbol, price, dip_decision)

        if not self.deal.paused and not self.deal.paused_sell:
            last_filled = self.deal.orders[self.deal.filled_count - 1]
            suppress_take_profit = (
                self._last_stop_loss_result is not None
                and self._last_stop_loss_result.trailing_active
                and self.bot.dca_trailing_replaces_take_profit
            )
            # LONG take-profit triggers on the way UP to target; SHORT
            # mirrors it — target sits BELOW average, hit on the way DOWN.
            take_profit_hit = price <= last_filled.target if is_short else price >= last_filled.target
            if not suppress_take_profit and take_profit_hit:
                return await self._handle_sell(symbol, price, reason="take_profit")

        return _Tick.RETRY, TICK_INTERVAL_SEC

    # -- order handlers -------------------------------------------------------

    async def _handle_base_order(self, symbol: str, price: float) -> tuple[str, float]:
        # LONG opens by buying; SHORT opens by selling first (bought back
        # later in _handle_sell) — same mirroring as backtest.py/
        # dca_bot_manager.build_initial_orders.
        base = self.deal.orders[0]
        open_side = "Sell" if self.bot.side == "short" else "Buy"
        try:
            order = await self._call(self.exchange.place_market_order(symbol, open_side, base.qty))
            status = await self._call(self.exchange.verify_order(symbol, order.order_id))
        except (asyncio.TimeoutError, OSError):
            logger.warning("Base order placement/verify timed out or hit a network error — retrying next tick.")
            return _Tick.RETRY, RETRY_INTERVAL_SEC
        except InsufficientMarginError:
            # Not retryable the way a timeout is — balance won't recover
            # on its own within RETRY_INTERVAL_SEC, and digging further
            # into an already-exhausted account is exactly wrong. Pause
            # for visibility/manual reconcile instead of retrying forever.
            logger.warning("Base order needs more margin than is available — pausing this deal.")
            self.pause(True, reason="insufficient_margin")
            return _Tick.RETRY, RETRY_INTERVAL_SEC
        if status.status != "filled":
            self.pause(True, reason="order_verify_buy")
            return _Tick.RETRY, RETRY_INTERVAL_SEC

        # Update order with actual fill price from Bybit (live must enter at current price)
        actual_price = status.avg_price if status.avg_price > 0 else base.price
        self.deal.orders[0] = replace(base, filled=1, price=actual_price)
        self.deal.filled_count = 1
        self.deal.is_start = 1
        # Trailing-stop extreme starts at the entry fill, then tracks the
        # low (short) or high (long) as ticks arrive — mirrors
        # backtest.py's per-bar OHLC initialization, adapted for a
        # tick-driven (no bars) live/paper feed.
        self.deal.trail_high_price = actual_price
        # Funding is only ever charged for boundaries crossed AFTER the
        # deal opens — matches backtest.py's "skip any funding events
        # that already happened before this deal opened" behavior.
        self.deal.last_funding_settlement_ts = time.time()
        self.pause(False)
        return _Tick.RETRY, TICK_INTERVAL_SEC

    async def _handle_safety_order(self, symbol: str, price: float) -> bool:
        idx = self.deal.filled_count
        if idx >= len(self.deal.orders):
            return False
        rung = self.deal.orders[idx]
        is_short = self.bot.side == "short"
        # LONG safety orders sit BELOW entry, triggered as price falls to
        # or below the rung; SHORT sits ABOVE entry, triggered as price
        # rises to or above it.
        rung_triggered = price >= rung.price if is_short else price <= rung.price
        if rung.filled or not rung_triggered:
            return False

        add_side = "Sell" if is_short else "Buy"
        try:
            order = await self._call(self.exchange.place_market_order(symbol, add_side, rung.qty))
            status = await self._call(self.exchange.verify_order(symbol, order.order_id))
        except (asyncio.TimeoutError, OSError):
            logger.warning("Safety order placement/verify timed out or hit a network error — retrying next tick.")
            return False
        except InsufficientMarginError:
            logger.warning("Safety order needs more margin than is available — pausing this deal.")
            self.pause(True, pause_buy=True, reason="insufficient_margin")
            return False
        if status.status != "filled":
            self.pause(True, pause_buy=True, reason="order_verify_buy")
            return False

        # Update order with actual fill price from Bybit (live must enter at current price)
        actual_price = status.avg_price if status.avg_price > 0 else rung.price
        self.deal.orders[idx] = replace(rung, filled=1, price=actual_price)
        self.deal.filled_count = idx + 1
        self.pause(False)
        return True

    async def _compute_reversal_qty(self, symbol: str, price: float) -> float:
        """Sizes the reversal's NEW position from CURRENT account
        balance — never mirrors the qty of the position being closed.
        That qty is an accident of how deep the DCA ladder got (base +
        however many safety orders happened to fill), not a deliberate
        size for a fresh directional bet; a real stop-and-reverse system
        always re-sizes the reversal the same way any brand-new position
        would be sized, using whatever risk budget exists right now.
        Mirrors dca_bot_manager.size_bot_to_available_funds's single-rung
        math directly (importing that module here would be circular —
        it imports DCABotEngine from this one).
        """
        leverage = self.bot.leverage if self.bot.leverage > 0 else 1.0
        margin = self.bot.first_order_amount
        if self.bot.auto_size_to_funds:
            balances = await self.exchange.get_balance("USDT")
            available = balances.get("USDT", 0.0)
            target_budget = max(available, 0.0) * (self.bot.funds_utilization_percent / 100)
            margin, _ = dca_math.solve_order_sizing_for_budget(
                self.bot.first_order_amount, self.bot.dca_order_amount, self.bot.dca_max_order,
                self.bot.dca_order_size_multiplier, self.bot.exchange_fee, target_budget,
            )
        notional = margin * leverage
        qty_raw = notional / price if price else 0.0
        precision = await self.exchange.get_precision(symbol)
        return self.exchange.filter_amount(precision, qty_raw)

    async def _handle_sell(self, symbol: str, price: float, reason: str) -> tuple[str, float]:
        # LONG closes by selling; SHORT closes by buying back to cover.
        last_filled = self.deal.orders[self.deal.filled_count - 1]
        close_side = "Buy" if self.bot.side == "short" else "Sell"

        # Close and open the opposite position in one order for drawdown
        # reversals. Other exit reasons close flat.
        reverse_qty = 0.0
        if reason == "reverse_drawdown":
            reverse_qty = await self._compute_reversal_qty(symbol, price)

        try:
            order = await self._call(
                self.exchange.place_market_order(symbol, close_side, last_filled.qty_sum + reverse_qty)
            )
            status = await self._call(self.exchange.verify_order(symbol, order.order_id))
        except (asyncio.TimeoutError, OSError):
            logger.warning("Closing order placement/verify timed out or hit a network error — retrying next tick.")
            return _Tick.RETRY, RETRY_INTERVAL_SEC
        except InsufficientMarginError:
            if reverse_qty > 0:
                # The flip's extra margin isn't available — fall back to
                # a PLAIN close instead of leaving a drawdown-triggered
                # exit unexecuted indefinitely: closing the actual open
                # position is the safety-critical action here, the
                # reversal is opportunistic on top of it.
                logger.warning("Reversal flip needs more margin than available — closing flat instead.")
                reverse_qty = 0.0
                try:
                    order = await self._call(
                        self.exchange.place_market_order(symbol, close_side, last_filled.qty_sum)
                    )
                    status = await self._call(self.exchange.verify_order(symbol, order.order_id))
                except (asyncio.TimeoutError, OSError, InsufficientMarginError):
                    logger.warning("Fallback close also failed — pausing this deal.")
                    self.pause(True, pause_sell=True, reason="insufficient_margin")
                    return _Tick.RETRY, RETRY_INTERVAL_SEC
            else:
                # Only reachable via place_market_order's flip-excess path
                # on a plain (non-reversing) close, which this bot never
                # sends — defense-in-depth, not an expected path.
                logger.warning("Closing order needs more margin than is available — pausing this deal.")
                self.pause(True, pause_sell=True, reason="insufficient_margin")
                return _Tick.RETRY, RETRY_INTERVAL_SEC
        if status.status != "filled":
            self.pause(True, pause_sell=True, reason="order_verify_sell")
            return _Tick.RETRY, RETRY_INTERVAL_SEC

        exit_price = status.avg_price or price
        # Amount rounding here is only cosmetic (a display figure), so no
        # exchange precision filter is needed for this profit estimate.
        profit = dca_math.calculate_profit(
            exit_price, last_filled.average, last_filled.sum,
            self.bot.dca_take_profit_percent, self.bot.exchange_fee, 0.0,
            filter_amount=lambda q: q, side=self.bot.side, leverage=self.bot.leverage,
        )
        # Net accumulated real funding into the result ONCE, here at
        # close — mirrors backtest.py's model exactly (funding_cost_accum
        # is subtracted directly into profit_quote at trade close, never
        # applied as a running mid-trade balance mutation). calculate_profit
        # itself stays funding-unaware (out of scope to change), so the
        # adjustment happens here and profit_percent is recomputed on the
        # same margin-based-ROI basis calculate_profit itself uses.
        leverage_for_margin = self.bot.leverage if self.bot.leverage > 0 else 1.0
        margin = last_filled.sum / leverage_for_margin
        adjusted_profit_quote = profit.current_profit_quote - self.deal.funding_cost_quote
        adjusted_profit_percent = round((adjusted_profit_quote / margin * 100) if margin else 0.0, 2)
        apply_funding = getattr(self.exchange, "apply_funding_cost", None)
        if apply_funding is not None:
            apply_funding(self.deal.funding_cost_quote)
        # Tiny-win classification, mirroring backtest.py's BacktestTrade.
        # is_real_win() so a paper/live deal's "win" label means the same
        # thing a backtest's does — same raw-move formula, same
        # MIN_WIN_PRICE_PCT price-move floor, same WIN_FEE_MULT fee-multiple
        # check. Both MIN_WIN_PRICE_PCT and raw_move_percent are in the same
        # percent-NUMBER convention (e.g. 0.33 means "0.33%"), so no unit
        # conversion needed here — see MIN_WIN_PRICE_PCT's own definition
        # in backtest.py for why that matters (this file used to carry a
        # local `* 100` workaround for a unit bug that has since been fixed
        # at the source; removed now that the source constant is correct).
        is_short = self.bot.side == "short"
        raw_move = (last_filled.average - exit_price) if is_short else (exit_price - last_filled.average)
        raw_move_percent = (raw_move / last_filled.average * 100) if last_filled.average else 0.0
        estimated_fees_quote = last_filled.sum * (self.bot.exchange_fee * 2 / 100)
        is_real_win = (
            adjusted_profit_quote > 0
            and raw_move_percent >= MIN_WIN_PRICE_PCT
            and not (estimated_fees_quote > 0 and adjusted_profit_quote <= WIN_FEE_MULT * estimated_fees_quote)
        )
        self.deal.sell_data = {
            "date": time.time(),
            "price": exit_price,
            "average": last_filled.average,
            "profit_percent": adjusted_profit_percent,
            "profit_quote": adjusted_profit_quote,
            "raw_move_percent": raw_move_percent,
            "estimated_fees_quote": estimated_fees_quote,
            "funding_cost_quote": self.deal.funding_cost_quote,
            "is_real_win": is_real_win,
            "reason": reason,
        }
        self.deal.status = DealStatus.CLOSED
        self.deal.date_closed = time.time()
        self.deal.canceled = reason == "cancel"
        self.deal.panic_sell = reason == "panic_sell"
        self.deal.stop_loss = reason == "stop_loss"
        self.deal.liquidated = reason == "liquidated"
        # Streak bookkeeping for the reverse-drawdown hysteresis: any
        # reversal-triggered close extends the streak; literally any
        # other close reason (a real win, a plain stop-loss, cancel,
        # panic, liquidation) means the strategy resolved normally —
        # reset it, so a later, unrelated drawdown starts a fresh count
        # rather than inheriting an old streak from a different move.
        if reason == "reverse_drawdown":
            self.bot.consecutive_reversals += 1
            self.bot.last_reversal_ts = time.time()
        else:
            self.bot.consecutive_reversals = 0

        # Record DIP analysis outcome if applicable
        if self.current_dip_record and self.calibration:
            try:
                bounced = reason == "take_profit"
                max_depth = abs((self.current_dip_record.entry_price - price) / self.current_dip_record.entry_price * 100) if self.current_dip_record.entry_price else 0
                recovery_candles = int((time.time() - self.current_dip_record.timestamp) / 2)  # ~2 sec per candle
                safeties_used = len([o for o in self.deal.orders[1:self.deal.filled_count] if not o.price == self.deal.orders[0].price])
                pnl_percent = (exit_price - last_filled.average) / last_filled.average * 100 if self.bot.side == "long" else (last_filled.average - exit_price) / last_filled.average * 100
                pnl_account = pnl_percent * (self.bot.leverage if self.bot.leverage > 0 else 1)

                self.calibration.record_outcome(
                    dip_record=self.current_dip_record,
                    bounced=bounced,
                    max_depth_percent=max_depth,
                    recovery_candles=recovery_candles,
                    safety_orders_needed=safeties_used,
                    pnl=pnl_account,
                )
                # Feed the SAME offline models (XGBoost outcome predictor,
                # RL decision optimizer) that only ever learned from
                # forward-tester simulations before — a real live/paper
                # deal closing never reached trade_memory.json at all.
                # Schema matches what load_all_forward_test_trades()
                # already produces (see trade_memory.py's
                # append_live_paper_trade docstring for the exact keys
                # each model reads).
                dims = self.current_dip_record.dimension_scores or {}
                patterns = self.current_dip_record.patterns_matched or []
                append_live_paper_trade({
                    "timestamp": datetime.now().isoformat(),
                    "was_win": bounced,
                    "pnl_pct": pnl_account,
                    "volatility": dims.get("volatility", 1.5),
                    "momentum": dims.get("momentum", 50.0),
                    "pattern_success_rate": dims.get("patterns", 50.0),
                    "action_taken": self.current_dip_record.decision_action,
                    "price_change_pct": pnl_percent,
                    "dip_depth_pct": self.current_dip_record.dip_depth_percent,
                    "confidence": self.current_dip_record.decision_confidence,
                    "leverage": self.bot.leverage if self.bot.leverage > 0 else 1,
                    "candles_since_entry": recovery_candles,
                    "pattern": patterns[0] if patterns else "unknown",
                    "source": "live_paper",
                })
                self.current_dip_record = None
                self.dip_service.reset_dip_state()
            except Exception as e:
                logger.warning(f"Failed to record DIP outcome: {e}")

        if reverse_qty > 0:
            # Tell the manager a new position already exists (opened by
            # the SAME order as this close) — it must register a
            # monitoring engine for it, not place a fresh order.
            opposite_side = "short" if self.bot.side == "long" else "long"
            self.deal.pending_flip = {"side": opposite_side, "fill_price": exit_price, "qty": reverse_qty}
        return _Tick.DONE, 0.0

    # -- dip analysis and profiting ------------------------------------------------

    async def _analyze_dip(self, symbol: str, price: float, ticker=None):
        """Analyze current price for bounce opportunity using DIP analysis."""
        if not self.deal.is_start or not self.dip_service:
            return None

        try:
            # First tick for entry: set entry price
            if self.dip_service.entry_price is None:
                last_filled = self.deal.orders[self.deal.filled_count - 1]
                self.dip_service.set_entry_price(last_filled.average)

            # Real current spread when this tick's ticker has one — see
            # dip_analysis_service.py's analyze_current_dip docstring for
            # why this falls back to an estimate instead when it doesn't.
            spread_bps = None
            if ticker is not None and ticker.bid and ticker.ask and ticker.last:
                spread_bps = (ticker.ask - ticker.bid) / ticker.last * 10_000

            # Analyze dip and record decision
            decision = await self.dip_service.analyze_current_dip(
                current_price=price,
                account_balance=10000.0,  # Placeholder, would get from exchange
                position_size=self.deal.orders[self.deal.filled_count - 1].qty_sum,
                leverage=self.bot.leverage if self.bot.leverage > 0 else 1,
                tp_percent=self.bot.dca_take_profit_percent,
                bid_ask_spread_bps=spread_bps,
            )

            if decision and self.calibration:
                # Record decision for calibration
                dip_depth = abs(
                    (price - self.dip_service.entry_price) / self.dip_service.entry_price * 100
                ) if self.dip_service.entry_price else 0

                if dip_depth > 0.3:  # Only log meaningful dips
                    # decision.matched_patterns/dimension_scores are the
                    # REAL BounceAnalysis output now (see DrawdownDecision's
                    # docstring) — this used to reach into
                    # self.dip_service.dip_service.analyzer.last_analysis,
                    # an attribute path that never existed on
                    # DipAnalysisService (hasattr always False), silently
                    # keeping dimension_scores empty and patterns_matched
                    # empty for every trade ever recorded here. It also
                    # read decision.recommendation, a field DrawdownDecision
                    # never had — an AttributeError swallowed by this
                    # method's own broad except below, so this whole block
                    # had likely never successfully recorded a single
                    # calibration entry.
                    self.current_dip_record = self.calibration.record_decision(
                        entry_price=self.dip_service.entry_price,
                        dip_depth_percent=dip_depth,
                        decision_confidence=decision.confidence,
                        decision_action=decision.recommendation,
                        patterns_matched=decision.matched_patterns,
                        dimension_scores=decision.dimension_scores,
                        source="live_paper",
                    )

            return decision
        except Exception as e:
            logger.warning(f"DIP analysis error: {e}")
            return None

    async def _place_dip_safety_order(self, symbol: str, price: float, decision):
        """Place a safety order based on DIP analysis decision."""
        try:
            last_filled = self.deal.orders[self.deal.filled_count - 1]
            safety_size = last_filled.qty_sum * (decision.safety_amount / 100)

            if safety_size <= 0:
                return

            is_short = self.bot.side == "short"
            add_side = "Sell" if is_short else "Buy"

            logger.info(
                f"DIP SAFETY ORDER | Conf: {decision.confidence:.0f}% | "
                f"Size: {safety_size:.8f} | Est Time: {decision.time_estimate_candles} candles"
            )

            try:
                order = await self._call(self.exchange.place_market_order(symbol, add_side, safety_size))
                status = await self._call(self.exchange.verify_order(symbol, order.order_id))
            except (asyncio.TimeoutError, OSError):
                logger.warning("DIP safety order placement/verify timed out — retrying next tick.")
                return
            except InsufficientMarginError:
                logger.warning("DIP safety order needs more margin — pausing this deal.")
                self.pause(True, pause_buy=True, reason="insufficient_margin")
                return

            if status.status != "filled":
                logger.warning("DIP safety order not filled — pausing for manual reconcile.")
                self.pause(True, pause_buy=True, reason="order_verify_buy")
                return

            actual_price = status.avg_price if status.avg_price > 0 else price
            logger.info(f"DIP safety order filled at {actual_price:.2f}")
            self.pause(False)

        except Exception as e:
            logger.warning(f"DIP safety order error: {e}")

    # -- stop-and-reverse ------------------------------------------------------

    def _evaluate_reverse_drawdown(self, price: float) -> bool:
        """Live per-tick check against the position's OWN drawdown —
        NOT triggered by any close event. Uses the same raw price-move
        profit_pct convention as _evaluate_stop_loss (not leverage-
        adjusted margin ROI), so reverse_drawdown_percent and
        dca_stop_loss_percent mean the same thing when compared side by
        side on the same BotConfig.

        Hysteresis against whipsaw thrash: a naive "flip the instant the
        threshold crosses" is exactly how a real account bleeds on fees/
        slippage when a sharp move bounces right back — price dips, this
        flips short, price immediately reverts, the new short is also
        underwater, it flips long again, and so on with no real edge.
        reverse_cooldown_sec enforces a minimum gap between reversals;
        max_consecutive_reversals caps how many can happen in a row
        before this just defers to the deal's own normal exits (stop-
        loss/take-profit/liquidation) instead of reversing again — a
        real win or a plain stop-loss close resets the streak (see
        _handle_sell), so this cap is about one sustained bad stretch,
        not a lifetime limit.
        """
        last_filled = self.deal.orders[self.deal.filled_count - 1]
        if not last_filled.average:
            return False
        is_short = self.bot.side == "short"
        profit_pct = (
            (last_filled.average - price) / last_filled.average * 100
            if is_short else (price - last_filled.average) / last_filled.average * 100
        ) - self.bot.exchange_fee
        if profit_pct > -self.bot.reverse_drawdown_percent:
            return False  # not underwater enough yet

        if time.time() - self.bot.last_reversal_ts < self.bot.reverse_cooldown_sec:
            return False
        if self.bot.consecutive_reversals >= self.bot.max_consecutive_reversals:
            return False
        return True

    # -- stop-loss / trailing --------------------------------------------------

    def _evaluate_stop_loss(self, price: float) -> bool:
        last_filled = self.deal.orders[self.deal.filled_count - 1]
        last_so_price = self.deal.orders[self.deal.filled_count - 1].price
        is_short = self.bot.side == "short"
        profit_pct = (
            (
                (last_filled.average - price) / last_filled.average * 100
                if is_short
                else (price - last_filled.average) / last_filled.average * 100
            )
            - self.bot.exchange_fee
            if last_filled.average
            else 0.0
        )
        # Trailing extreme: lowest price seen (short) or highest (long) —
        # mirrors backtest.py's tick loop exactly.
        self.deal.trail_high_price = (
            min(self.deal.trail_high_price, price) if is_short else max(self.deal.trail_high_price, price)
        )

        result = evaluate_stop_loss(
            StopLossInput(
                enabled=self.bot.dca_stop_loss_enabled,
                price=price,
                average=last_filled.average,
                stop_loss_percent=self.bot.dca_stop_loss_percent,
                reference=self.bot.dca_stop_loss_reference,  # type: ignore[arg-type]
                last_safety_order_price=last_so_price,
                fee_rate=self.bot.exchange_fee,
                move_breakeven=self.bot.dca_stop_loss_move_breakeven,
                breakeven_trigger=self.bot.dca_stop_loss_breakeven_trigger,
                profit_percentage=profit_pct,
                breakeven_armed=self.deal.stop_loss_breakeven_armed,
                active_stop_loss_price=self.deal.active_stop_loss_price,
                trailing_enabled=self.bot.dca_trailing_stop_enabled,
                trailing_distance=self.bot.dca_trailing_stop_distance,
                trailing_activate_profit=self.bot.dca_trailing_activate_profit,
                trail_high_price=self.deal.trail_high_price,
                side=self.bot.side,
            )
        )
        self._last_stop_loss_result = result
        if result.breakeven_armed:
            self.deal.stop_loss_breakeven_armed = True
        # Ratchet only ever tightens: for long the stop only ever rises,
        # for short it only ever falls — same as backtest.py's
        # ratchet_improves, including the "never set yet" 0.0 baseline case.
        ratchet_improves = self.deal.active_stop_loss_price == 0.0 or (
            result.level < self.deal.active_stop_loss_price
            if is_short
            else result.level > self.deal.active_stop_loss_price
        )
        if ratchet_improves and (result.breakeven_armed or result.trailing_active):
            self.deal.active_stop_loss_price = result.level
        return result.triggered
