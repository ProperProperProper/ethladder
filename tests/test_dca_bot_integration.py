import asyncio

import pytest

from symbot_python.exchange.base import (
    InstrumentPrecision,
    InsufficientMarginError,
    OrderResult,
    OrderStatus,
    Ticker,
    TradingMode,
)
from symbot_python.strategy import dca_bot
from symbot_python.strategy.dca_bot_manager import DCABotManager
from symbot_python.strategy.models import BotConfig, Deal, DealStatus


class FakeExchangeClient:
    """Test double implementing the ExchangeClient protocol with a
    scripted price feed and instant-fill order simulation. Used ONLY in
    this test file to drive the engine's decision logic deterministically
    without hitting a real API — it is never imported by application
    code. The two real implementations the running bot actually uses are
    exchange/bybit_client.py (live orders via pybit) and
    exchange/paper_client.py (real live Bybit market data, simulated
    fills only).
    """

    mode = TradingMode.PAPER

    def __init__(
        self, price_sequence: list[float], precision: InstrumentPrecision | None = None,
        balance: float = 1_000_000.0,
    ):
        self._prices = list(price_sequence)
        self._served = 0
        self._precision = precision or InstrumentPrecision(
            symbol="BTCUSDT", tick_size=0.01, qty_step=0.0001, min_order_qty=0.0001, min_order_amt=1.0
        )
        self._orders: dict[str, OrderStatus] = {}
        self._next_id = 0
        self.placed: list[tuple[str, str, float, float]] = []  # (symbol, side, qty, fill_price)
        self.balance = balance
        self.leverage_calls: list[tuple[str, float]] = []

    def _next_price(self) -> float:
        idx = min(self._served, len(self._prices) - 1)
        self._served += 1
        return self._prices[idx]

    async def get_precision(self, symbol, force_refresh=False):
        return self._precision

    def filter_price(self, precision, price):
        return price

    def filter_amount(self, precision, qty):
        return qty

    async def get_ticker(self, symbol):
        price = self._next_price()
        return Ticker(symbol=symbol, last=price, bid=price, ask=price, volume_24h_base=0, turnover_24h_quote=0)

    async def get_kline(self, symbol, interval, limit=200):
        return []

    async def place_market_order(self, symbol, side, qty):
        self._next_id += 1
        order_id = f"fake-{self._next_id}"
        fill_price = self._prices[min(self._served - 1, len(self._prices) - 1)]
        self.placed.append((symbol, side, qty, fill_price))
        self._orders[order_id] = OrderStatus(
            order_id=order_id, status="filled", avg_price=fill_price,
            cum_exec_qty=qty, cum_exec_value=qty * fill_price, cum_exec_fee=0.0,
        )
        return OrderResult(order_id=order_id, symbol=symbol, side=side, qty_requested=qty)

    async def get_order_status(self, symbol, order_id):
        return self._orders.get(
            order_id,
            OrderStatus(order_id=order_id, status="unknown", avg_price=0, cum_exec_qty=0, cum_exec_value=0, cum_exec_fee=0),
        )

    async def verify_order(self, symbol, order_id):
        return await self.get_order_status(symbol, order_id)

    async def get_balance(self, coin=None):
        if coin and coin != "USDT":
            return {}
        return {"USDT": self.balance}

    async def ensure_leverage(self, symbol, leverage):
        self.leverage_calls.append((symbol, leverage))

    async def get_maintenance_margin_rate(self, symbol):
        return 0.005


def make_bot(**overrides) -> BotConfig:
    defaults = dict(
        bot_name="test-bot",
        pair="BTC/USDT",
        first_order_amount=100.0,
        dca_order_amount=50.0,
        dca_max_order=1,
        dca_order_size_multiplier=1.0,
        dca_order_start_distance=5.0,
        dca_order_step_percent=5.0,
        dca_order_step_percent_multiplier=1.0,
        dca_take_profit_percent=2.0,
        exchange_fee=0.0,
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


class AlwaysMarginRejectingExchangeClient(FakeExchangeClient):
    """place_market_order always raises InsufficientMarginError —
    simulates an account that can never open this deal's base order.
    Counts calls so a test can prove the engine actually STOPS retrying
    once paused, rather than hammering place_market_order forever.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.place_order_calls = 0

    async def place_market_order(self, symbol, side, qty):
        self.place_order_calls += 1
        raise InsufficientMarginError("simulated: never enough margin")


class CrashingExchangeClient(FakeExchangeClient):
    """get_ticker raises a plain exception (not a timeout) on exactly the
    `crash_on_call_number`-th call (1-indexed), then behaves normally
    forever after — reproduces a one-off engine crash from a genuine
    bug/unexpected error, as opposed to a stalled network call or a
    permanently-broken exchange. Without _run_engine_and_recover, this
    kills the task completely silently (no log, deal frozen forever,
    Cancel/Panic also dead since they route through the same now-defunct
    tick loop).
    """

    def __init__(self, *args, crash_on_call_number: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._crash_on_call_number = crash_on_call_number
        self._call_number = 0

    async def get_ticker(self, symbol):
        self._call_number += 1
        if self._call_number == self._crash_on_call_number:
            raise RuntimeError("simulated unexpected engine crash")
        return await super().get_ticker(symbol)


class MarginRejectingExchangeClient(FakeExchangeClient):
    """place_market_order raises InsufficientMarginError on exactly the
    `raise_on_call_number`-th call (1-indexed) — reproduces a real
    account-level rejection (e.g. a safety order that would need more
    margin than remains) rather than a network problem. Real regression:
    PaperExchangeClient used to debit margin_balance below zero with no
    floor at all; this exercises the caller's handling of the guard that
    now raises instead.
    """

    def __init__(self, *args, raise_on_call_number: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._raise_on_call_number = raise_on_call_number
        self._call_number = 0

    async def place_market_order(self, symbol, side, qty):
        self._call_number += 1
        if self._call_number == self._raise_on_call_number:
            raise InsufficientMarginError("simulated insufficient margin")
        return await super().place_market_order(symbol, side, qty)


class HangingThenRecoveringExchangeClient(FakeExchangeClient):
    """get_ticker hangs forever on one call (simulating a stalled real
    network call) after `calls_before_hang` normal calls, then behaves
    normally again — reproduces the exact real bug this guards against:
    a live/paper deal that stopped responding to Cancel/Panic sell with
    nothing in the logs, because a single stuck exchange call blocked
    the engine's tick loop from ever getting back around to checking
    those flags.
    """

    def __init__(self, *args, calls_before_hang: int = 0, hang_calls: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls_before_hang = calls_before_hang
        self._hang_calls_remaining = hang_calls

    async def get_ticker(self, symbol):
        if self._calls_before_hang > 0:
            self._calls_before_hang -= 1
        elif self._hang_calls_remaining > 0:
            self._hang_calls_remaining -= 1
            await asyncio.sleep(3600)  # never resolves within any test timeout
        return await super().get_ticker(symbol)


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    # Real engine ticks sleep 1-2s; tests use a small but NON-ZERO delay.
    # A literal 0.0 makes asyncio.sleep a bare cooperative yield rather
    # than a real timer-queue entry, so a whole chain of ticks (e.g. a
    # deal closing and immediately auto-chaining the next one) can run to
    # completion in a single scheduling burst, racing straight past
    # whatever intermediate state a test's poll-based wait_until is
    # trying to observe. A small real delay puts every tick through the
    # same timer heap as the test's polling sleeps, so state transitions
    # stay observable one at a time.
    monkeypatch.setattr(dca_bot, "TICK_INTERVAL_SEC", 0.005)
    monkeypatch.setattr(dca_bot, "RETRY_INTERVAL_SEC", 0.002)
    monkeypatch.setattr(dca_bot, "EXCHANGE_CALL_TIMEOUT_SEC", 0.05)


@pytest.fixture
def managers():
    created: list[DCABotManager] = []
    yield created


@pytest.fixture(autouse=True)
async def _cleanup_managers(managers):
    yield
    for manager in managers:
        await manager.stop()


async def wait_until(condition, timeout=2.0, interval=0.002):
    async def poll():
        while not condition():
            await asyncio.sleep(interval)

    await asyncio.wait_for(poll(), timeout=timeout)


async def test_base_order_then_take_profit_closes_profitably(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])  # isolate this test from auto-chain
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.sell_data["reason"] == "take_profit"
    assert deal.sell_data["profit_percent"] > 0
    assert exchange.placed[0][1] == "Buy"  # base order
    assert exchange.placed[-1][1] == "Sell"


async def test_short_base_order_then_take_profit_closes_profitably(managers):
    # Mirror of test_base_order_then_take_profit_closes_profitably: a
    # short opens by SELLING, profits as price FALLS, and closes by
    # BUYING back to cover — this exact path was completely broken until
    # dca_bot.py/dca_bot_manager.py were made side-aware (previously
    # every order was hardcoded Buy-to-open/Sell-to-close regardless of
    # bot.side, so a "short" bot was silently trading long the whole time).
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 97.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], side="short")
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.sell_data["reason"] == "take_profit"
    assert deal.sell_data["profit_percent"] > 0
    assert exchange.placed[0][1] == "Sell"  # opens the short
    assert exchange.placed[-1][1] == "Buy"  # buys back to close


async def test_short_safety_order_triggers_before_take_profit(managers):
    # Mirror of test_safety_order_triggers_before_take_profit: a short's
    # safety order sits ABOVE entry (price rising against it triggers
    # adding), not below.
    # Leading 100.0 x3: deal-start consumes 2 ticker calls before the
    # engine's own tick loop starts (analyze_direction's OMLX check, then
    # the ticker build_initial_orders is sized against) — the tick loop's
    # own first call is what actually fills the base order.
    exchange = FakeExchangeClient([100.0, 100.0, 100.0, 106.0, 90.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], side="short")
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.filled_count == 2  # base + 1 safety order
    assert deal.sell_data["reason"] == "take_profit"
    sells = [p for p in exchange.placed if p[1] == "Sell"]
    assert len(sells) == 2  # base + safety order, both add to the short


async def test_multiple_safety_orders_all_fill_across_successive_ticks_long(managers):
    # backtest.py can fill several safety-order rungs within ONE candle
    # (a big bar-range move crosses multiple trigger prices at once); the
    # live/paper engine only ever sees one discrete tick price at a time,
    # so it can only fill one rung per tick. This proves that difference
    # doesn't silently DROP a rung — a big move still fills every
    # qualifying rung, just one tick later than the next, not never.
    # dca_max_order=2, start_distance=5%, step=5% (flat, multiplier=1.0):
    # rung1 @ 95, rung2 @ 90 (deviation(2) = 2*5 = 10%).
    # Leading 100.0 x3: see test_short_safety_order_triggers_before_take_profit's comment.
    exchange = FakeExchangeClient([100.0, 100.0, 100.0, 89.0, 89.0, 130.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_max_order=2)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.filled_count == 3  # base + BOTH safety orders, none skipped
    assert deal.sell_data["reason"] == "take_profit"
    buys = [p for p in exchange.placed if p[1] == "Buy"]
    assert len(buys) == 3


async def test_multiple_safety_orders_all_fill_across_successive_ticks_short(managers):
    # Mirror of the long version above: rung1 @ 105, rung2 @ 110 (short
    # safety orders sit above entry).
    # Leading 100.0 x3: see test_short_safety_order_triggers_before_take_profit's comment.
    exchange = FakeExchangeClient([100.0, 100.0, 100.0, 111.0, 111.0, 50.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], side="short", dca_max_order=2)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.filled_count == 3  # base + BOTH safety orders, none skipped
    assert deal.sell_data["reason"] == "take_profit"
    sells = [p for p in exchange.placed if p[1] == "Sell"]
    assert len(sells) == 3


async def test_sell_data_marks_a_clear_profitable_close_as_a_real_win(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])  # TP=2.0%, exchange_fee=0.0
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.sell_data["is_real_win"] is True
    assert deal.sell_data["raw_move_percent"] > 0.33
    assert deal.sell_data["estimated_fees_quote"] == pytest.approx(0.0)


async def test_sell_data_marks_a_sub_033_percent_move_as_not_a_real_win(managers):
    # Mirrors backtest.py's test_backtest_win_rate_excludes_a_marginal_
    # take_profit_close: a genuinely tiny (<0.33%) raw move must not
    # count as a real win even though profit_quote > 0. TP=0.1% -> target
    # = 100.1, reached exactly, giving a 0.1% raw move — under the floor.
    exchange = FakeExchangeClient([100.0, 100.0, 100.1])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_take_profit_percent=0.1)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.sell_data["profit_quote"] > 0  # genuinely profitable in raw dollars
    assert deal.sell_data["raw_move_percent"] < 0.33
    assert deal.sell_data["is_real_win"] is False


async def test_short_stop_loss_closes_deal_at_a_loss(managers):
    # Mirror of test_stop_loss_closes_deal_at_a_loss: for a short, a
    # stop-loss triggers on a price RISE against the position, not a fall.
    exchange = FakeExchangeClient([100.0, 100.0, 106.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], side="short",
        dca_stop_loss_enabled=True, dca_stop_loss_percent=5.0, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.stop_loss is True
    assert deal.sell_data["reason"] == "stop_loss"
    assert deal.sell_data["profit_percent"] < 0


async def test_short_cancel_deal_closes_it_via_market_buy(managers):
    # The exact real-world scenario this closes: a short paper deal that
    # previously could never be cancelled/panic-sold correctly, because
    # closing always placed another "Sell" (extending the short or
    # erroring) instead of the "Buy" needed to actually cover it.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 100.6, 100.6, 100.6])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], side="short", dca_take_profit_percent=1000.0)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED)
    assert deal.canceled is True
    assert deal.sell_data["reason"] == "cancel"
    assert exchange.placed[0][1] == "Sell"   # opened the short
    assert exchange.placed[-1][1] == "Buy"   # cancel closes it by covering


async def test_safety_order_triggers_before_take_profit(managers):
    # Leading 100.0 x3: see test_short_safety_order_triggers_before_take_profit's comment.
    exchange = FakeExchangeClient([100.0, 100.0, 100.0, 94.0, 105.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.filled_count == 2  # base + 1 safety order
    assert deal.sell_data["reason"] == "take_profit"
    buys = [p for p in exchange.placed if p[1] == "Buy"]
    assert len(buys) == 2


async def test_stop_loss_closes_deal_at_a_loss(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 94.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"],
        dca_stop_loss_enabled=True, dca_stop_loss_percent=5.0, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.stop_loss is True
    assert deal.sell_data["reason"] == "stop_loss"
    assert deal.sell_data["profit_percent"] < 0


async def test_implausible_price_spike_is_held_not_acted_on(managers):
    # tick sequence: entry, base-fill, IMPLAUSIBLE spike, recovery, take-profit
    exchange = FakeExchangeClient([100.0, 100.0, 250.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    # exactly base buy + final sell — the implausible tick placed no order
    assert len(exchange.placed) == 2
    assert deal.sell_data["reason"] == "take_profit"


async def test_can_start_deal_blocks_second_deal_on_same_pair(managers):
    exchange = FakeExchangeClient([100.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot()
    manager.add_bot(bot)
    manager.deals["existing"] = Deal(bot_id=bot.bot_id, pair=bot.pair, orders=[], status=DealStatus.ACTIVE)

    ok, reason = manager.can_start_deal(bot)
    assert ok is False
    assert reason == "pair_already_active"


async def test_can_start_deal_blocked_by_circuit_breaker(managers):
    exchange = FakeExchangeClient([100.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot()
    manager.add_bot(bot)
    manager.circuit_breaker_active = True

    ok, reason = manager.can_start_deal(bot)
    assert ok is False
    assert reason == "circuit_breaker_active"


async def test_circuit_breaker_prevents_new_deal_from_actually_starting(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot()
    manager.add_bot(bot)
    manager.circuit_breaker_active = True
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await asyncio.sleep(0.05)  # give the queue consumer a chance to (not) act
    assert len(manager.deals) == 0
    assert exchange.placed == []


async def test_auto_chain_starts_a_new_deal_after_completion(managers):
    # Deal 1 closes via take-profit; deal 2 then starts with entry price
    # clamped to the sequence's last value (103.0) and sits open forever
    # (never reaches its own target or safety price) — a deliberately
    # simple, stable two-deal end state. Leading 100.0 x3: see
    # test_short_safety_order_triggers_before_take_profit's comment.
    exchange = FakeExchangeClient([100.0, 100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(deal_cool_down=0)  # default start_conditions=["asap"]
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 2, timeout=3.0)
    statuses = [d.status for d in manager.deals.values()]
    assert statuses.count(DealStatus.CLOSED) == 1
    assert statuses.count(DealStatus.ACTIVE) == 1
    assert bot.deal_count == 2


async def test_reverse_drawdown_flips_into_the_opposite_side_when_underwater(managers):
    # Regression target: the reversal must trigger off the position's
    # OWN drawdown, checked live every tick — NOT off a take_profit
    # close (a win is not "a drawdown to take advantage of"). Long entry
    # ~100.0, reverse_drawdown_percent=1.5; price dips to 98.0 (a 2% move)
    # on tick 3, well before its absurdly-high take_profit could ever fire.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 98.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], reverse_drawdown_percent=1.5,
        auto_size_to_funds=False, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    first_deal = next(iter(manager.deals.values()))
    await wait_until(lambda: first_deal.status == DealStatus.CLOSED, timeout=2.0)
    assert first_deal.sell_data["reason"] == "reverse_drawdown"

    # The closing order itself must be sized bigger than what was held —
    # that's what actually drives place_market_order's flip branch,
    # rather than two separate orders.
    closing_order = exchange.placed[-1]
    assert closing_order[1] == "Sell"  # closes a long
    assert closing_order[2] > first_deal.orders[0].qty  # sized to close AND reopen, not just close

    await wait_until(lambda: len(manager.deals) == 2, timeout=2.0)
    flipped_deal = next(d for d in manager.deals.values() if d.deal_id != first_deal.deal_id)
    assert flipped_deal.status == DealStatus.ACTIVE
    assert flipped_deal.is_start == 1
    assert flipped_deal.filled_count == 1
    flipped_engine = manager.engines[flipped_deal.deal_id]
    assert flipped_engine.bot.side == "short"  # opposite of the original long
    # Entry price for the new side is the actual flip fill price, not a
    # freshly-fetched ticker — it was already determined by the same order.
    assert flipped_deal.orders[0].price == pytest.approx(first_deal.sell_data["price"])
    assert bot.consecutive_reversals == 1


async def test_reverse_drawdown_falls_back_to_a_plain_close_when_the_flip_cant_be_margined(managers):
    # call #1 = base order (tick 1, price 100.0); call #2 = the combined
    # close+flip attempt (tick 3, price 98.0 crosses the drawdown
    # threshold) — rejected, forcing the fallback to a plain (smaller) close.
    exchange = MarginRejectingExchangeClient([100.0, 100.0, 100.5, 98.0], raise_on_call_number=2)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], reverse_drawdown_percent=1.5,
        auto_size_to_funds=False, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)

    # The combined (flip) order was rejected, but the deal still closes
    # normally — a drawdown-triggered exit must not be left unexecuted
    # just because the bonus reversal couldn't be margined.
    assert deal.sell_data["reason"] == "reverse_drawdown"
    assert deal.pending_flip is None
    # The retried, plain-sized close is what actually filled.
    assert exchange.placed[-1][2] == pytest.approx(deal.orders[0].qty)
    # No second deal — nothing to flip into.
    await asyncio.sleep(0.05)
    assert len(manager.deals) == 1


async def test_reverse_drawdown_never_triggers_on_cancel_panic_or_liquidation(managers):
    # Explicit user exits and blown positions must never auto-reopen
    # anything, even with reverse_drawdown_percent set.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 100.5])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], reverse_drawdown_percent=1.5, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)
    assert deal.sell_data["reason"] == "cancel"
    assert deal.pending_flip is None
    await asyncio.sleep(0.05)
    assert len(manager.deals) == 1  # no flip, no second deal


async def test_take_profit_and_plain_stop_loss_never_trigger_a_reversal(managers):
    # A WIN is not "a drawdown to take advantage of" — confirms
    # reverse_drawdown_percent has zero effect on an ordinary profitable
    # take_profit close, even though it's technically also a "close".
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], reverse_drawdown_percent=50.0)  # absurdly high: never crosses
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)
    assert deal.sell_data["reason"] == "take_profit"
    assert deal.pending_flip is None
    await asyncio.sleep(0.05)
    assert len(manager.deals) == 1


async def test_reverse_drawdown_respects_the_cooldown_between_reversals(managers):
    # After one reversal fires, a second real drawdown crossing on the
    # newly-flipped deal must NOT reverse again within reverse_cooldown_sec.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 98.0, 98.0, 100.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], reverse_drawdown_percent=1.5, reverse_cooldown_sec=3600.0,
        auto_size_to_funds=False, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    first_deal = next(iter(manager.deals.values()))
    await wait_until(lambda: first_deal.status == DealStatus.CLOSED, timeout=2.0)
    await wait_until(lambda: len(manager.deals) == 2, timeout=2.0)
    flipped_deal = next(d for d in manager.deals.values() if d.deal_id != first_deal.deal_id)

    # The flipped (short) deal now faces its own real adverse move
    # (price index 5 = 100.0, ~2% against a short entered near 98.0) —
    # cooldown (3600s, effectively infinite in test time) must block it.
    await asyncio.sleep(0.15)
    assert flipped_deal.status == DealStatus.ACTIVE
    assert bot.consecutive_reversals == 1  # unchanged — no second reversal happened


async def test_reverse_drawdown_stops_after_max_consecutive_reversals(managers):
    # Same real second adverse move as the cooldown test, but with
    # cooldown disabled and the streak cap set to 1 — isolates the CAP
    # specifically as what blocks the second reversal.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 98.0, 98.0, 100.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(
        start_conditions=["api"], reverse_drawdown_percent=1.5, reverse_cooldown_sec=0.0,
        max_consecutive_reversals=1, auto_size_to_funds=False, dca_take_profit_percent=1000.0,
    )
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    first_deal = next(iter(manager.deals.values()))
    await wait_until(lambda: first_deal.status == DealStatus.CLOSED, timeout=2.0)
    await wait_until(lambda: len(manager.deals) == 2, timeout=2.0)
    flipped_deal = next(d for d in manager.deals.values() if d.deal_id != first_deal.deal_id)

    await asyncio.sleep(0.15)
    assert flipped_deal.status == DealStatus.ACTIVE  # cap (not cooldown) blocked the second reversal
    assert bot.consecutive_reversals == 1


async def test_many_auto_chained_deals_do_not_leak_memory_or_engines(managers):
    # Long-run resilience smoke test: paper trading is meant to run
    # unattended for weeks, auto-chaining a new deal every time one
    # closes. Simulates 50 open/take-profit-close cycles in a row and
    # asserts nothing grows unbounded — MAX_CLOSED_DEALS_PER_BOT pruning
    # keeps manager.deals bounded, and manager.engines never accumulates
    # a stale entry from a deal that already finished.
    # Per cycle: analyze_direction's OMLX check, _try_start_deal's own
    # sizing ticker, the engine's base-order-fill tick, then the
    # take-profit tick — see test_short_safety_order_triggers_before_take_profit's comment.
    prices = [100.0, 100.0, 100.0, 103.0] * 70  # comfortably more than 50 cycles' worth
    exchange = FakeExchangeClient(prices)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(deal_cool_down=0)  # default start_conditions=["asap"] -> auto-chains
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: bot.deal_count >= 50, timeout=10.0)

    # Never more than the single currently-in-flight deal's engine.
    assert len(manager.engines) <= 1
    assert len(manager._tasks) <= 1
    # Pruned to the retention cap, not accumulating one entry per cycle
    # forever (would be 50+ here without pruning).
    from symbot_python.strategy.dca_bot_manager import MAX_CLOSED_DEALS_PER_BOT
    assert len(manager.deals) <= MAX_CLOSED_DEALS_PER_BOT + 1


async def test_api_mode_bot_never_auto_chains(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    await wait_until(lambda: _sole_deal_closed(manager))
    await asyncio.sleep(0.05)  # let any (incorrect) auto-chain attempt happen
    assert len(manager.deals) == 1


def _sole_deal_closed(manager: DCABotManager) -> bool:
    if not manager.deals:
        return False
    return next(iter(manager.deals.values())).status == DealStatus.CLOSED


async def test_cancel_deal_closes_it_via_market_sell(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 100.6, 100.6, 100.6])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_take_profit_percent=1000.0)  # keep it open until we cancel
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)  # base order filled

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED)
    assert deal.canceled is True
    assert deal.sell_data["reason"] == "cancel"


async def test_panic_sell_deal_closes_it_immediately(managers):
    exchange = FakeExchangeClient([100.0, 100.0, 90.0, 90.0, 90.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_take_profit_percent=1000.0)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)

    manager.panic_sell_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED)
    assert deal.panic_sell is True
    assert deal.sell_data["reason"] == "panic_sell"


async def test_a_stalled_exchange_call_times_out_instead_of_freezing_the_deal(managers):
    # calls_before_hang=2 lets deal-creation (_try_start_deal's own
    # get_ticker) and the engine's first tick (which fills the base
    # order) succeed normally — the hang lands on the FIRST tick after
    # the deal is already open and "monitoring", exactly the real
    # scenario: an already-running deal that stopped responding.
    exchange = HangingThenRecoveringExchangeClient(
        [100.0, 100.0, 100.5, 100.6, 100.6, 100.6, 100.6, 100.6],
        calls_before_hang=2, hang_calls=1,
    )
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_take_profit_percent=1000.0)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)

    # Pre-fix, the tick that hits the stalled get_ticker() call would
    # never return, so the loop would never come back around to notice
    # this cancel request — it would sit "monitoring" forever.
    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)
    assert deal.canceled is True


async def test_an_engine_crash_force_closes_the_deal_and_auto_chains(managers):
    # call #1 = analyze_direction's OMLX check, #2 = _try_start_deal's own
    # ticker fetch (entry price for build_initial_orders), #3 = the
    # engine's first tick (fills the base order), #4 = the engine's next
    # tick — crashes there, exactly like an already-"monitoring" deal
    # hitting an unexpected error. Pre-fix this kills the task with zero
    # visibility and the deal sits frozen forever, unresponsive to
    # everything including Cancel/Panic — the exact bug reproduced live.
    exchange = CrashingExchangeClient([100.0, 100.0, 100.0, 100.5, 100.5, 100.5, 100.5], crash_on_call_number=4)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot()  # active, auto-chaining bot (no start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    first_deal = next(iter(manager.deals.values()))
    await wait_until(lambda: first_deal.is_start == 1)

    # The crash happens on the tick right after — must force-close with a
    # distinct reason, not sit frozen.
    await wait_until(lambda: first_deal.status == DealStatus.CLOSED, timeout=2.0)
    assert first_deal.sell_data["reason"] == "engine_error"

    # And the bot must keep working — a replacement deal auto-chains
    # rather than the whole bot silently dying with its one deal.
    await wait_until(lambda: len(manager.deals) == 2, timeout=2.0)
    assert len(manager.engines) <= 1  # the crashed engine was cleaned up, not leaked


async def test_a_safety_order_rejected_for_insufficient_margin_pauses_not_crashes(managers):
    # call #1 = base order placement (first tick, price 100.0); call #2 =
    # the safety order placement triggered by the price 94.0 tick — that
    # one is rejected. Pre-fix this had no distinct handling at all: an
    # uncaught InsufficientMarginError would propagate past the
    # (TimeoutError, OSError) catch straight to _run_engine_and_recover,
    # force-closing a deal that should instead just pause for visibility.
    exchange = MarginRejectingExchangeClient([100.0, 100.0, 94.0, 105.0], raise_on_call_number=2)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)  # base order filled fine

    await wait_until(lambda: deal.paused is True, timeout=2.0)
    assert deal.pause_reason == "insufficient_margin"
    assert deal.status != DealStatus.CLOSED  # paused, not force-closed
    assert deal.filled_count == 1  # the rejected safety order never filled


async def test_a_permanently_rejected_base_order_stops_retrying_once_paused(managers):
    # Regression: is_start stays 0 until the base order fills, and
    # _tick()'s dispatch to _handle_base_order was unconditional on
    # is_start==0 alone — unlike the safety-order/take-profit paths,
    # nothing gated it on deal.paused. Pre-fix, a permanently-rejected
    # base order hammered place_market_order (and the exchange) forever
    # at RETRY_INTERVAL_SEC cadence instead of actually halting.
    exchange = AlwaysMarginRejectingExchangeClient([100.0] * 20)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.paused is True, timeout=2.0)
    assert deal.pause_reason == "insufficient_margin"

    calls_at_pause = exchange.place_order_calls
    await asyncio.sleep(0.2)  # many tick intervals at the test's fast_ticks speed
    # Bounded, not growing without limit — a small amount of slack covers
    # a call that was already in flight the instant pause() took effect.
    assert exchange.place_order_calls <= calls_at_pause + 1


async def test_cancel_closes_a_deal_stuck_paused_before_its_base_order_ever_filled(managers):
    # Same bug, the other half: a deal stuck at is_start==0 must still be
    # cancellable. Pre-fix, Cancel/Panic requests were only ever checked
    # further down in _tick(), past the unconditional is_start==0 early
    # return — meaning a stuck-at-open deal was completely unresponsive
    # to Cancel/Panic, the exact class of bug already fixed once this
    # session for a stalled network call, reappearing via a different
    # code path.
    exchange = AlwaysMarginRejectingExchangeClient([100.0] * 20)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.paused is True, timeout=2.0)
    assert deal.is_start == 0  # never opened

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)
    assert deal.canceled is True
    assert deal.sell_data["reason"] == "cancel"
    assert deal.sell_data["profit_quote"] == 0.0  # nothing was ever risked


class FundingAwareFakeExchangeClient(FakeExchangeClient):
    """Adds the funding-simulation surface PaperExchangeClient implements
    (get_current_funding_rate/apply_funding_cost) on top of the base
    FakeExchangeClient — used to prove dca_bot.py actually accrues and
    nets real funding cost the same way backtest.py's funding_events
    does, closing the gap where paper trading would otherwise look
    systematically more profitable than backtest for the identical
    strategy, purely because funding was never simulated at all.
    """

    def __init__(self, *args, funding_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.funding_rate = funding_rate
        self.funding_cost_calls: list[float] = []

    async def get_current_funding_rate(self, symbol):
        return self.funding_rate

    def apply_funding_cost(self, cost_quote: float) -> None:
        self.funding_cost_calls.append(cost_quote)


async def test_long_deal_pays_funding_when_rate_is_positive(managers, monkeypatch):
    # Real FUNDING_INTERVAL_SEC (8h) can't be waited out in a test —
    # shrink it so several boundaries pass while this deal sits open.
    monkeypatch.setattr(dca_bot, "FUNDING_INTERVAL_SEC", 0.05)
    exchange = FundingAwareFakeExchangeClient([100.0] * 300, funding_rate=0.001)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], dca_take_profit_percent=1000.0)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)
    await asyncio.sleep(0.3)  # let several shrunken funding boundaries pass while open

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)

    assert deal.funding_cost_quote > 0  # long PAYS when rate > 0
    # Netted into margin_balance exactly once, at close, with the deal's
    # total accrued cost — never applied as a running mid-trade mutation.
    assert exchange.funding_cost_calls == [pytest.approx(deal.funding_cost_quote)]
    assert deal.sell_data["funding_cost_quote"] == pytest.approx(deal.funding_cost_quote)


async def test_short_deal_receives_funding_when_rate_is_positive(managers, monkeypatch):
    monkeypatch.setattr(dca_bot, "FUNDING_INTERVAL_SEC", 0.05)
    exchange = FundingAwareFakeExchangeClient([100.0] * 300, funding_rate=0.001)
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], side="short", dca_take_profit_percent=1000.0)
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.is_start == 1)
    await asyncio.sleep(0.3)

    manager.cancel_deal(deal.deal_id)
    await wait_until(lambda: deal.status == DealStatus.CLOSED, timeout=2.0)

    assert deal.funding_cost_quote < 0  # short RECEIVES (mirror) when rate > 0
    assert exchange.funding_cost_calls == [pytest.approx(deal.funding_cost_quote)]


async def test_no_funding_accrues_when_exchange_does_not_support_it(managers):
    # FakeExchangeClient (no get_current_funding_rate) must not crash the
    # tick loop — soft feature-detection no-ops cleanly.
    exchange = FakeExchangeClient([100.0, 100.0, 100.5, 103.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"])
    manager.add_bot(bot)
    await manager.start()

    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    await wait_until(lambda: deal.status == DealStatus.CLOSED)

    assert deal.funding_cost_quote == 0.0


def test_clear_circuit_breaker(managers):
    exchange = FakeExchangeClient([100.0])
    manager = DCABotManager(exchange)
    managers.append(manager)
    manager.circuit_breaker_active = True

    manager.clear_circuit_breaker()
    assert manager.circuit_breaker_active is False


async def test_funding_accrues_correctly_across_multiple_boundaries_in_one_gap():
    # Direct unit test of _accrue_funding_if_due (bypassing the full tick
    # loop for precision): if the gap between checks spans exactly 3
    # funding boundaries (e.g. the process was busy/paused for a while),
    # the accrued cost must be exactly 3x a single boundary's charge, at
    # the CURRENT rate (fetched once, not once per boundary) — not 1x
    # (missed crossings) and not fetched-and-summed differently per
    # boundary in a way that could double count.
    from symbot_python.strategy.dca_bot import DCABotEngine, FUNDING_INTERVAL_SEC
    from symbot_python.strategy.dca_math import OrderRung

    exchange = FundingAwareFakeExchangeClient([100.0], funding_rate=0.001)
    bot = make_bot(side="long")
    filled_rung = OrderRung(price=100.0, qty=10.0, amount=1000.0, qty_sum=10.0, sum=1000.0, average=100.0, target=102.0, filled=1)
    deal = Deal(bot_id=bot.bot_id, pair=bot.pair, orders=[filled_rung], filled_count=1, is_start=1)

    engine = DCABotEngine(bot, deal, exchange)
    boundary = FUNDING_INTERVAL_SEC
    deal.last_funding_settlement_ts = 10 * boundary  # sits exactly on a boundary

    # Jump forward exactly 3 boundaries in one gap (simulates a long pause
    # between ticks, not 3 separate calls).
    import symbot_python.strategy.dca_bot as dca_bot_module
    fake_now = 13 * boundary + 1.0  # 3 boundaries later, safely past the 3rd
    orig_time = dca_bot_module.time.time
    dca_bot_module.time.time = lambda: fake_now
    try:
        await engine._accrue_funding_if_due(price=100.0)
    finally:
        dca_bot_module.time.time = orig_time

    expected_single_boundary_cost = filled_rung.qty_sum * 100.0 * 0.001  # notional * rate
    assert deal.funding_cost_quote == pytest.approx(expected_single_boundary_cost * 3)
    assert deal.last_funding_settlement_ts == pytest.approx(fake_now)


async def test_sized_reversal_preserves_cooldown_and_streak(managers):
    exchange = FakeExchangeClient([100, 100, 98])
    manager = DCABotManager(exchange)
    managers.append(manager)
    bot = make_bot(start_conditions=["api"], auto_size_to_funds=True,
                   reverse_drawdown_percent=1.5, reverse_cooldown_sec=3600,
                   max_consecutive_reversals=1, dca_take_profit_percent=1000)
    manager.add_bot(bot)
    await manager.start()
    await manager.request_deal_start(bot.bot_id)
    await wait_until(lambda: len(manager.deals) == 2)
    flipped = next(d for d in manager.deals.values() if d.status == DealStatus.ACTIVE)
    engine = manager.engines[flipped.deal_id]
    assert engine.bot.consecutive_reversals == 1
    assert bot.consecutive_reversals == 1
    assert not engine._evaluate_reverse_drawdown(102)
    actual_qty = exchange.placed[1][2] - exchange.placed[0][2]
    assert flipped.orders[0].qty_sum == pytest.approx(actual_qty)
