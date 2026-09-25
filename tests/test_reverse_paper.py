import pytest

from symbot_python.exchange.paper_client import PaperExchangeClient
from symbot_python.exchange.reverse_paper import ReversePaper
from tests.test_bybit_client import FakeSession


@pytest.mark.asyncio
async def test_mirrors_opposite_fill_and_close_with_separate_wallet():
    original = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1000})
    reverse = ReversePaper(original, 1000)
    await original.ensure_leverage("BTCUSDT", 10)

    opened = await original.place_market_order("BTCUSDT", "Buy", 1)
    assert original.position("BTCUSDT").qty == 1
    assert reverse.client.position("BTCUSDT").qty == -1
    assert reverse.fills[0].source_order_id == opened.order_id
    assert reverse.fills[0].side == "Sell"
    # The original Buy crossed the spread at the ask (100.6); the mirrored
    # opposite-side Sell must fill at the bid (100.4) — the price a real
    # simultaneous Sell would actually get, not the same price the Buy
    # paid (that would give the reverse account a free, zero-spread fill
    # on every mirrored trade).
    assert reverse.fills[0].source_price == pytest.approx(100.6)
    assert reverse.fills[0].mirror_price == pytest.approx(100.4)

    await original.place_market_order("BTCUSDT", "Sell", 1)
    assert original.position("BTCUSDT").qty == 0
    assert reverse.client.position("BTCUSDT").qty == 0
    assert len(reverse.fills) == 2
    assert reverse.fills[1].side == "Buy"
    assert reverse.client is not original


@pytest.mark.asyncio
async def test_rejected_original_fill_is_not_mirrored():
    original = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1})
    reverse = ReversePaper(original, 1)
    with pytest.raises(Exception):
        await original.place_market_order("BTCUSDT", "Buy", 1)
    assert reverse.fills == []


@pytest.mark.asyncio
async def test_reverse_failure_does_not_break_original_fill():
    original = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1000})
    reverse = ReversePaper(original, 1)
    result = await original.place_market_order("BTCUSDT", "Buy", 1)
    assert result.order_id
    assert original.position("BTCUSDT").qty == 1
    assert reverse.client.position("BTCUSDT").qty == 0
    assert reverse.fills[0].error is not None


@pytest.mark.asyncio
async def test_reverse_funding_is_opposite_and_independent():
    original = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1000})
    reverse = ReversePaper(original, 1000)
    original.apply_funding_cost(2.5)
    assert original.margin_balance == 997.5
    assert reverse.client.margin_balance == 1002.5
