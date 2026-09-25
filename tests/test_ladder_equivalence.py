"""Proves build_initial_orders (dca_bot_manager.py — the live/paper order
ladder builder) produces an IDENTICAL order ladder to build_ladder
(backtest.py) given equivalent inputs. This is the actual guarantee
behind "paper trades exactly like the backtest says it should live" —
if these two ever diverge, a config that looked safe/profitable in the
backtest could behave differently once it's actually running.
"""

import asyncio

import pytest

from symbot_python.exchange.base import InstrumentPrecision
from symbot_python.strategy.backtest import BacktestConfig, build_ladder
from symbot_python.strategy.dca_bot_manager import build_initial_orders
from symbot_python.strategy.models import BotConfig

PRICE_TICK = 0.01
QTY_STEP = 0.0001


class RealRoundingFakeExchange:
    """Unlike test_dca_bot_integration.py's FakeExchangeClient (identity
    filters, for testing decision logic in isolation), this uses REAL
    round_to_step rounding — the same function backtest.py's own
    filter_price/filter_amount use — so a ladder-construction comparison
    against build_ladder is fair or the fake would flatter one side.
    """

    def __init__(self):
        self._precision = InstrumentPrecision(
            symbol="ETHUSDT", tick_size=PRICE_TICK, qty_step=QTY_STEP,
            min_order_qty=QTY_STEP, min_order_amt=1.0,
        )

    async def get_precision(self, symbol, force_refresh=False):
        return self._precision

    def filter_price(self, precision, price):
        from symbot_python.exchange.utils import round_to_step
        return round_to_step(price, precision.tick_size)

    def filter_amount(self, precision, qty):
        from symbot_python.exchange.utils import round_to_step
        return round_to_step(qty, precision.qty_step)


DCA_PARAMS = dict(
    first_order_amount=100.0,
    dca_order_amount=50.0,
    dca_max_order=5,
    dca_order_size_multiplier=1.08,
    dca_order_start_distance=1.3,
    dca_order_step_percent=1.3,
    dca_order_step_percent_multiplier=1.05,
    dca_take_profit_percent=1.5,
    exchange_fee=0.06,
)

ENTRY_PRICE = 2500.1234


def assert_ladders_match(backtest_orders, live_orders):
    assert len(backtest_orders) == len(live_orders)
    for i, (bt, live) in enumerate(zip(backtest_orders, live_orders)):
        assert bt.price == pytest.approx(live.price, rel=1e-9), f"rung {i} price"
        assert bt.qty == pytest.approx(live.qty, rel=1e-9), f"rung {i} qty"
        assert bt.amount == pytest.approx(live.amount, rel=1e-9), f"rung {i} amount"
        assert bt.qty_sum == pytest.approx(live.qty_sum, rel=1e-9), f"rung {i} qty_sum"
        assert bt.sum == pytest.approx(live.sum, rel=1e-9), f"rung {i} sum"
        assert bt.average == pytest.approx(live.average, rel=1e-9), f"rung {i} average"
        assert bt.target == pytest.approx(live.target, rel=1e-9), f"rung {i} target"


@pytest.mark.parametrize("side,leverage", [
    ("long", 1.0), ("long", 11.0), ("short", 1.0), ("short", 11.0),
])
def test_build_initial_orders_matches_build_ladder(side, leverage):
    backtest_config = BacktestConfig(
        **DCA_PARAMS, side=side, leverage=leverage,
        price_tick=PRICE_TICK, min_move_amount=QTY_STEP,
    )
    backtest_orders = build_ladder(backtest_config, ENTRY_PRICE)

    bot = BotConfig(bot_name="equiv-test", pair="ETH/USDT", side=side, leverage=leverage, **DCA_PARAMS)
    exchange = RealRoundingFakeExchange()
    live_orders = asyncio.run(build_initial_orders(bot, exchange, "ETHUSDT", ENTRY_PRICE))

    assert_ladders_match(backtest_orders, live_orders)
