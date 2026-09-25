"""Bot/deal lifecycle orchestration for ONE exchange client (i.e. one
trading-mode instance — paper or live; backtest never goes through here
at all, see strategy/backtest.py). Deals are held in memory for now —
DB persistence is Phase 2; the field names in strategy/models.py are
already chosen to map onto that schema without renaming.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Optional

from symbot_python.exchange.base import ExchangeClient
from symbot_python.strategy import dca_math
from symbot_python.strategy.dca_bot import DCABotEngine
from symbot_python.strategy.models import BotConfig, Deal, DealStatus, to_exchange_symbol
from symbot_python.strategy.signal_bot import is_api_start
from symbot_python.strategy.omlx_direction import analyze_direction

logger = logging.getLogger(__name__)

# In-memory retention cap per bot. Paper state is already ephemeral (no DB
# persistence yet — restarting the server clears everything, per this
# module's own docstring), so pruning old CLOSED deals just bounds how
# much history one long-running, auto-chaining bot accumulates before
# `self.deals` — and every /paper page render, which iterates ALL of a
# bot's deals on every load — grows without limit over weeks/months of
# unattended uptime. Active deals are never pruned, only closed ones past
# the cap, oldest first.
MAX_CLOSED_DEALS_PER_BOT = 200


async def build_initial_orders(
    bot: BotConfig, exchange: ExchangeClient, symbol: str, entry_price: float
) -> list[dca_math.OrderRung]:
    """Compute the full static order ladder for a deal starting at
    entry_price — computed once at deal creation and anchored to the
    base order's ask price, rather than recomputed on the fly as the
    market moves.

    bot.first_order_amount/dca_order_amount are MARGIN. When
    bot.leverage > 1, each rung's actual notional (what qty/amount/
    average/target are computed from) is margin * leverage — "same
    margin, N x bigger position." Mirrors strategy/backtest.py's
    build_ladder exactly — LONG: safety orders sit BELOW entry, target
    ABOVE average. SHORT is the mirror image: safety orders ABOVE entry,
    target BELOW average, same deviation percentages, opposite direction.
    """
    precision = await exchange.get_precision(symbol)
    leverage = bot.leverage if bot.leverage > 0 else 1.0
    is_short = bot.side == "short"
    direction = 1 if is_short else -1  # sign applied to the deviation %

    def filter_price(p: float) -> float:
        return exchange.filter_price(precision, p)

    def filter_amount(q: float) -> float:
        return exchange.filter_amount(precision, q)

    orders: list[dca_math.OrderRung] = []

    base_notional = bot.first_order_amount * leverage
    base_qty_raw = base_notional / entry_price
    base_adj = dca_math.calculate_adjustments(
        entry_price, base_qty_raw, bot.exchange_fee, precision.qty_step, filter_amount, filter_price
    )
    orders.append(
        dca_math.OrderRung(
            price=entry_price, qty=base_adj.order_qty, amount=base_adj.order_amount,
            qty_sum=0, sum=0, average=0, target=0, filled=0,
        )
    )

    prev_margin = bot.dca_order_amount
    for i in range(1, bot.dca_max_order + 1):
        if i == 1:
            price = filter_price(entry_price * (1 + direction * bot.dca_order_start_distance / 100))
            margin = bot.dca_order_amount
        else:
            deviation = dca_math.get_deviation_dca(
                bot.dca_order_step_percent, bot.dca_order_step_percent_multiplier, i
            )
            price = filter_price(entry_price * (1 + direction * deviation / 100))
            margin = prev_margin * bot.dca_order_size_multiplier
        notional = margin * leverage
        qty_raw = notional / price
        adj = dca_math.calculate_adjustments(
            price, qty_raw, bot.exchange_fee, precision.qty_step, filter_amount, filter_price
        )
        orders.append(
            dca_math.OrderRung(
                price=price, qty=adj.order_qty, amount=adj.order_amount,
                qty_sum=0, sum=0, average=0, target=0, filled=0,
            )
        )
        prev_margin = margin

    return dca_math.recalculate_orders(
        orders, None, bot.exchange_fee, precision.qty_step,
        bot.dca_take_profit_percent, filter_amount, filter_price, precision.tick_size,
        side=bot.side,
    )


async def size_bot_to_available_funds(bot: BotConfig, exchange: ExchangeClient) -> BotConfig:
    """Returns a copy of `bot` with first_order_amount/dca_order_amount
    rescaled (ratio preserved) so the full ladder's MARGIN requirement
    equals funds_utilization_percent of the exchange's current available
    balance. No-op if bot.auto_size_to_funds is False.
    """
    if not bot.auto_size_to_funds:
        return bot
    balances = await exchange.get_balance("USDT")
    available = balances.get("USDT", 0.0)
    target_budget = max(available, 0.0) * (bot.funds_utilization_percent / 100)
    sized_first, sized_dca = dca_math.solve_order_sizing_for_budget(
        bot.first_order_amount, bot.dca_order_amount, bot.dca_max_order,
        bot.dca_order_size_multiplier, bot.exchange_fee, target_budget,
    )
    return replace(bot, first_order_amount=sized_first, dca_order_amount=sized_dca)


class DCABotManager:
    def __init__(self, exchange: ExchangeClient):
        self.exchange = exchange
        self.bots: dict[str, BotConfig] = {}
        self.deals: dict[str, Deal] = {}
        self.engines: dict[str, DCABotEngine] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        self._start_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queue_task: Optional[asyncio.Task] = None

        self.circuit_breaker_active = False

    # -- lifecycle -------------------------------------------------------------

    def add_bot(self, bot: BotConfig) -> None:
        self.bots[bot.bot_id] = bot

    async def start(self) -> None:
        self._queue_task = asyncio.create_task(self._consume_start_queue())

    async def stop(self) -> None:
        if self._queue_task:
            self._queue_task.cancel()
        for engine in self.engines.values():
            engine.request_stop()
        for task in list(self._tasks.values()):
            task.cancel()

    @property
    def active(self) -> bool:
        """Satisfies the CircuitBreaker protocol DCABotEngine expects."""
        return self.circuit_breaker_active

    # -- deal-start gating (§2c of the strategy spec) ---------------------------

    def active_deals_for_pair(self, pair: str) -> list[Deal]:
        return [d for d in self.deals.values() if d.pair == pair and d.status == DealStatus.ACTIVE]

    def can_start_deal(self, bot: BotConfig) -> tuple[bool, str]:
        if self.circuit_breaker_active:
            return False, "circuit_breaker_active"

        active_for_pair = self.active_deals_for_pair(bot.pair)

        if bot.pair_bots_deals_max and len(active_for_pair) >= bot.pair_bots_deals_max:
            return False, "pair_bots_deals_max"

        if bot.pair_deals_max > 1:
            if len(active_for_pair) >= bot.pair_deals_max:
                return False, "pair_deals_max"
        elif active_for_pair:
            return False, "pair_already_active"

        if bot.deal_max and bot.deal_count >= bot.deal_max:
            return False, "deal_max_reached"

        return True, "ok"

    async def request_deal_start(self, bot_id: str) -> None:
        """The single entry point for starting a deal, regardless of
        trigger source (asap loop, manual API call, signal, auto-chain).
        Serialized through one queue consumer so two simultaneous
        triggers can never both pass can_start_deal and open two deals.
        """
        await self._start_queue.put(bot_id)

    async def _consume_start_queue(self) -> None:
        while True:
            bot_id = await self._start_queue.get()
            try:
                await self._try_start_deal(bot_id)
            except Exception:
                logger.exception("deal start failed for bot_id=%s", bot_id)

    async def _try_start_deal(self, bot_id: str) -> None:
        bot = self.bots.get(bot_id)
        if not bot or not bot.active:
            return
        ok, _reason = self.can_start_deal(bot)
        if not ok:
            return

        symbol = to_exchange_symbol(bot.pair)

        # Query OMLX to determine safe direction (long/short/wait)
        direction = await analyze_direction(self.exchange, symbol, bot.side)
        if direction == "wait":
            logger.info("OMLX says wait, skipping deal start this cycle")
            return
        if direction in ("long", "short"):
            # Use OMLX's direction, which may override bot.side. Mutated
            # in place (BotConfig is a plain, non-frozen dataclass, same
            # as every other bot.<field> = ... elsewhere in this class) —
            # NOT via dataclasses.replace(), which would return a new
            # object and silently detach this local `bot` from
            # self.bots[bot_id]. That was a real bug here previously:
            # `bot.deal_count += 1` below mutated an orphaned copy,
            # self.bots[bot_id] stayed at deal_count=0 forever whenever
            # OMLX returned a concrete direction (the normal case), so
            # deal_max caps never actually triggered and anything reading
            # manager.bots[bot_id] (dashboard, tests) saw a permanently
            # stale count.
            bot.side = direction

        await self.exchange.ensure_leverage(symbol, bot.leverage if bot.leverage > 0 else 1.0)
        sized_bot = await size_bot_to_available_funds(bot, self.exchange)

        mmr = 0.0
        if sized_bot.leverage > 1:
            mmr = await self.exchange.get_maintenance_margin_rate(symbol)

        ticker = await self.exchange.get_ticker(symbol)
        orders = await build_initial_orders(sized_bot, self.exchange, symbol, ticker.last)
        deal = Deal(bot_id=bot_id, pair=bot.pair, orders=orders)
        self.deals[deal.deal_id] = deal
        bot.deal_count += 1

        engine = DCABotEngine(
            sized_bot, deal, self.exchange, on_complete=self._on_deal_complete,
            circuit_breaker=self, maintenance_margin_rate=mmr,
        )
        self.engines[deal.deal_id] = engine
        self._tasks[deal.deal_id] = asyncio.create_task(self._run_engine_and_recover(engine, deal))

    async def _run_engine_and_recover(self, engine: DCABotEngine, deal: Deal) -> None:
        """Wraps engine.run() so an unhandled exception can never leave a
        deal frozen forever with zero visibility. Without this, a crash
        inside the tick loop kills the task silently — Python only ever
        surfaces an "exception was never retrieved" warning when the Task
        object is garbage-collected, and self._tasks/self.engines hold a
        permanent reference to it until _on_deal_complete runs, which
        never happens on a crash. Observed directly: a real paper deal
        stopped responding to its own take-profit AND to Cancel/Panic
        (both of which also route through the now-dead tick loop) with
        nothing in the logs to explain why.

        On a crash: log the full traceback immediately (not deferred to
        GC), force-close the deal with a distinct reason so it's visibly
        different from a normal close in the UI, and still run the usual
        completion/auto-chain path so one bad deal doesn't permanently
        stall the whole bot.
        """
        try:
            await engine.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Deal %s's engine crashed and would otherwise be stuck forever — "
                "force-closing and auto-chaining a replacement.", deal.deal_id,
            )
            if deal.status != DealStatus.CLOSED:
                deal.status = DealStatus.CLOSED
                deal.date_closed = time.time()
                deal.sell_data = {
                    "date": deal.date_closed, "price": None, "average": None,
                    "profit_percent": 0.0, "profit_quote": 0.0, "reason": "engine_error",
                }
            await self._on_deal_complete(deal)

    # -- deal completion / auto-chain (§2i) -------------------------------------

    async def _on_deal_complete(self, deal: Deal) -> None:
        self._tasks.pop(deal.deal_id, None)
        self.engines.pop(deal.deal_id, None)
        self._prune_old_deals(deal.bot_id)

        bot = self.bots.get(deal.bot_id)
        if not bot:
            return

        if deal.config is not None:
            bot.consecutive_reversals = deal.config.consecutive_reversals
            bot.last_reversal_ts = deal.config.last_reversal_ts

        if deal.pending_flip is not None:
            # The closing order ALSO opened this position, in the same
            # atomic exchange call (see dca_bot.py's reverse_drawdown
            # handling) — it already exists regardless of active/
            # api-mode/cooldown, so it always needs a monitoring engine
            # or it would sit open with no stop-loss/take-profit ever
            # evaluated against it again.
            await self._start_deal_from_flip(bot, deal.pending_flip, deal.config)
            return

        if not bot.active:
            return
        if is_api_start(bot.start_conditions):
            return  # api-mode (Signal Bot) bots never auto-reopen

        if bot.deal_cool_down:
            await asyncio.sleep(bot.deal_cool_down)
        await self.request_deal_start(bot.bot_id)

    async def _start_deal_from_flip(self, bot: BotConfig, pending_flip: dict, position_config: Optional[BotConfig] = None) -> None:
        """Registers the position a reversal already opened with a fresh
        monitoring engine — the exchange-side fill already happened as
        part of the SAME order that closed the previous deal (see
        dca_bot.py's _handle_sell/_compute_reversal_qty, which sizes the
        reversal from current balance BEFORE sending that order), so
        this builds the new deal's order ladder around that already-
        known fill instead of placing a fresh one.
        first_order_amount is reverse-derived from the real filled qty
        so the rest of the safety-order ladder scales proportionally
        from the deal's actual base size, not an independently-guessed one.
        """
        symbol = to_exchange_symbol(bot.pair)
        opposite_side = pending_flip["side"]
        fill_price = pending_flip["fill_price"]
        fill_qty = pending_flip["qty"]
        source = position_config or bot
        leverage = source.leverage if source.leverage > 0 else 1.0
        implied_first_order_amount = (fill_qty * fill_price) / leverage
        scale = implied_first_order_amount / source.first_order_amount if source.first_order_amount else 1.0
        flip_bot = replace(source, side=opposite_side, first_order_amount=implied_first_order_amount,
                           dca_order_amount=source.dca_order_amount * scale)

        mmr = 0.0
        if flip_bot.leverage > 1:
            mmr = await self.exchange.get_maintenance_margin_rate(symbol)

        orders = await build_initial_orders(flip_bot, self.exchange, symbol, fill_price)
        precision = await self.exchange.get_precision(symbol)
        # This quantity has already traded. Never gross it up for fees again.
        orders[0] = replace(
            orders[0], qty=fill_qty, amount=fill_qty * fill_price,
            qty_sum=fill_qty, sum=fill_qty * fill_price, average=fill_price,
            target=dca_math.calculate_target_price(
                fill_price, flip_bot.dca_take_profit_percent, flip_bot.exchange_fee,
                lambda p: self.exchange.filter_price(precision, p), precision.tick_size,
                side=opposite_side,
            ), filled=1, manual=True,
        )
        orders = dca_math.recalculate_orders(
            orders, None, flip_bot.exchange_fee, precision.qty_step,
            flip_bot.dca_take_profit_percent,
            lambda q: self.exchange.filter_amount(precision, q),
            lambda p: self.exchange.filter_price(precision, p), precision.tick_size,
            side=opposite_side,
        )

        deal = Deal(
            bot_id=bot.bot_id, pair=bot.pair, orders=orders,
            filled_count=1, is_start=1, trail_high_price=fill_price,
            last_funding_settlement_ts=time.time(),
        )
        self.deals[deal.deal_id] = deal
        bot.deal_count += 1

        engine = DCABotEngine(
            flip_bot, deal, self.exchange, on_complete=self._on_deal_complete,
            circuit_breaker=self, maintenance_margin_rate=mmr,
        )
        self.engines[deal.deal_id] = engine
        self._tasks[deal.deal_id] = asyncio.create_task(self._run_engine_and_recover(engine, deal))

    def _prune_old_deals(self, bot_id: str) -> None:
        """Keeps only the most recent MAX_CLOSED_DEALS_PER_BOT CLOSED deals
        for this bot — active deals are never touched. See
        MAX_CLOSED_DEALS_PER_BOT's own comment for why this exists.
        """
        closed = sorted(
            (d for d in self.deals.values() if d.bot_id == bot_id and d.status == DealStatus.CLOSED),
            key=lambda d: d.date_opened, reverse=True,
        )
        for stale in closed[MAX_CLOSED_DEALS_PER_BOT:]:
            self.deals.pop(stale.deal_id, None)

    # -- manual controls, dispatched to the running engine ----------------------

    def cancel_deal(self, deal_id: str) -> None:
        engine = self.engines.get(deal_id)
        if engine:
            engine.request_cancel()

    def panic_sell_deal(self, deal_id: str) -> None:
        engine = self.engines.get(deal_id)
        if engine:
            engine.request_panic_sell()

    # -- circuit breaker: generic "halt all new deal starts" switch ------------
    # Nothing currently trips this automatically (the portfolio-realized-loss
    # checker that used to be the one caller was dead code — defined,
    # tested, never invoked from any real code path — and was removed).
    # The mechanism itself stays: can_start_deal() gates on it, the tick
    # loop respects it via the CircuitBreaker protocol, and it's manually
    # clearable from the paper trading UI — reusable if a real trip
    # condition is wired in later.

    def clear_circuit_breaker(self) -> None:
        self.circuit_breaker_active = False
