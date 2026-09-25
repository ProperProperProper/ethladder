import pytest

from symbot_python.exchange.base import InsufficientMarginError, TradingMode
from symbot_python.exchange.paper_client import PaperExchangeClient
from tests.test_bybit_client import FakeSession


@pytest.fixture
def client():
    return PaperExchangeClient(
        FakeSession(), initial_balances={"USDT": 1000.0}, fee_rate_percent=0.06
    )


def test_mode_is_paper(client):
    assert client.mode == TradingMode.PAPER


async def test_market_data_uses_real_session(client):
    ticker = await client.get_ticker("BTCUSDT")
    assert ticker.last == pytest.approx(100.5)
    precision = await client.get_precision("BTCUSDT")
    assert precision.tick_size == pytest.approx(0.01)


async def test_get_balance_returns_margin_asset(client):
    balances = await client.get_balance()
    assert balances == {"USDT": pytest.approx(1000.0)}


async def test_get_balance_filters_by_coin(client):
    assert await client.get_balance("USDT") == {"USDT": pytest.approx(1000.0)}
    assert await client.get_balance("BTC") == {}


async def test_ensure_leverage_is_per_symbol(client):
    await client.ensure_leverage("BTCUSDT", 11.0)
    assert client._leverage["BTCUSDT"] == 11.0
    # unaffected symbol defaults to 1.0 (no leverage) until set
    assert client._leverage.get("ETHUSDT", 1.0) == 1.0


async def test_buy_at_leverage_one_debits_full_notional_as_margin(client):
    await client.place_market_order("BTCUSDT", "Buy", 1.0)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(1.0)
    assert pos.avg_price == pytest.approx(100.6)  # fills at ask
    # margin = notional / leverage(=1) = 100.6; fee = 100.6*0.0006
    expected_fee = 100.6 * 0.0006
    assert client.margin_balance == pytest.approx(1000.0 - 100.6 - expected_fee)
    assert pos.margin_committed == pytest.approx(100.6)


async def test_buy_at_leverage_11x_debits_only_1_11th_as_margin(client):
    await client.ensure_leverage("BTCUSDT", 11.0)
    await client.place_market_order("BTCUSDT", "Buy", 11.0)  # 11 units notional = 11*100.6
    pos = client.position("BTCUSDT")
    notional = 11.0 * 100.6
    expected_margin = notional / 11.0
    assert pos.margin_committed == pytest.approx(expected_margin)
    expected_fee = notional * 0.0006
    assert client.margin_balance == pytest.approx(1000.0 - expected_margin - expected_fee)


async def test_second_buy_updates_vwap_average():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # fills at 100.6
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # fills at 100.6 again (FakeSession is static)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(2.0)
    assert pos.avg_price == pytest.approx(100.6)


async def test_sell_closes_position_and_credits_margin_plus_pnl(client):
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # margin committed ~100.6
    balance_after_buy = client.margin_balance
    order = await client.place_market_order("BTCUSDT", "Sell", 1.0)  # fills at bid 100.4
    status = await client.get_order_status("BTCUSDT", order.order_id)
    assert status.avg_price == pytest.approx(100.4)

    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(0.0)
    assert pos.margin_committed == pytest.approx(0.0)
    # realized pnl is negative here (sold below the buy fill price)
    realized_pnl = 1.0 * (100.4 - 100.6)
    fee = 100.4 * 0.0006
    assert client.margin_balance == pytest.approx(balance_after_buy + 100.6 + realized_pnl - fee)


async def test_partial_sell_only_closes_part_of_the_position():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("BTCUSDT", "Buy", 2.0)
    await client.place_market_order("BTCUSDT", "Sell", 1.0)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(1.0)
    assert pos.margin_committed > 0


async def test_verify_order_never_polls_always_instant(client):
    order = await client.place_market_order("BTCUSDT", "Buy", 1.0)
    status = await client.verify_order("BTCUSDT", order.order_id)
    assert status.status == "filled"


async def test_unknown_order_id_returns_unknown_status(client):
    status = await client.get_order_status("BTCUSDT", "nonexistent")
    assert status.status == "unknown"


async def test_sell_opens_a_short_when_flat(client):
    # Previously: a "Sell" against a flat position silently no-opped
    # (only a fee was charged, no position/margin recorded at all) —
    # the exact bug behind every short paper deal quietly losing money
    # and never actually opening a position.
    await client.place_market_order("BTCUSDT", "Sell", 1.0)  # fills at bid 100.4
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(-1.0)
    assert pos.avg_price == pytest.approx(100.4)
    assert pos.margin_committed == pytest.approx(100.4)
    expected_fee = 100.4 * 0.0006
    assert client.margin_balance == pytest.approx(1000.0 - 100.4 - expected_fee)


async def test_short_second_sell_updates_vwap_average():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("BTCUSDT", "Sell", 1.0)  # fills at 100.4
    await client.place_market_order("BTCUSDT", "Sell", 1.0)  # fills at 100.4 again
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(-2.0)
    assert pos.avg_price == pytest.approx(100.4)


async def test_buy_closes_a_short_and_credits_margin_plus_pnl(client):
    await client.place_market_order("BTCUSDT", "Sell", 1.0)  # opens short at bid 100.4
    balance_after_open = client.margin_balance
    order = await client.place_market_order("BTCUSDT", "Buy", 1.0)  # covers at ask 100.6
    status = await client.get_order_status("BTCUSDT", order.order_id)
    assert status.avg_price == pytest.approx(100.6)

    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(0.0)
    assert pos.margin_committed == pytest.approx(0.0)
    # short profits when it covers BELOW its entry — here it covers
    # ABOVE (100.6 > 100.4), so this is a real, negative realized loss.
    realized_pnl = 1.0 * (100.4 - 100.6)
    assert realized_pnl < 0
    fee = 100.6 * 0.0006
    assert client.margin_balance == pytest.approx(balance_after_open + 100.4 + realized_pnl - fee)


async def test_short_profits_when_price_falls():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("ETHUSDT", "Sell", 1.0)  # opens short at bid
    balance_after_open = client.margin_balance

    class LowerPriceSession(FakeSession):
        def get_tickers(self, **kwargs):
            return {
                "retCode": 0, "retMsg": "OK",
                "result": {"list": [{"lastPrice": "90", "bid1Price": "89.9", "ask1Price": "90.1"}]},
            }

    client._session = LowerPriceSession()
    await client.place_market_order("ETHUSDT", "Buy", 1.0)  # covers at the new, lower ask
    # Price fell after opening the short -> covering should be profitable,
    # not the money-losing no-op the pre-fix code produced for every short.
    assert client.margin_balance > balance_after_open


async def test_partial_buy_only_closes_part_of_a_short():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("BTCUSDT", "Sell", 2.0)
    await client.place_market_order("BTCUSDT", "Buy", 1.0)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(-1.0)
    assert pos.margin_committed > 0


async def test_order_exceeding_open_position_flips_to_the_opposite_side():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 100_000.0})
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # opens a 1.0 long
    await client.place_market_order("BTCUSDT", "Sell", 1.5)  # closes the long, flips to a 0.5 short
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(-0.5)
    assert pos.margin_committed > 0


async def test_a_flip_rejected_for_insufficient_margin_leaves_the_close_unapplied_too():
    # Regression: the flip-margin check used to run AFTER the close's own
    # mutations had already been applied, so a rejected flip left the
    # exchange in a torn state (old position closed, balance credited)
    # while the caller believed the whole order failed. Checking against
    # the balance the close WOULD produce, before mutating anything,
    # makes the whole order atomic: either both the close and the flip
    # happen, or neither does.
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 200.0})
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # opens a 1.0 long, leaves ~99 free
    balance_before = client.margin_balance
    pos_before_qty = client.position("BTCUSDT").qty

    with pytest.raises(InsufficientMarginError):
        # Closes the 1.0 long (credits ~100 back) then tries to flip the
        # remaining 2.0 into a new short, needing ~200 more margin than
        # the account has even after that credit.
        await client.place_market_order("BTCUSDT", "Sell", 3.0)

    assert client.margin_balance == pytest.approx(balance_before)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(pos_before_qty)  # the close never applied either


async def test_a_closing_loss_is_capped_at_the_positions_own_committed_margin():
    # Isolated margin: the most a position can ever lose is what was
    # actually committed to it — mirrors a real exchange force-
    # liquidating before losses reach into the rest of the account.
    # Regression: this used to apply the raw (uncapped) mark-to-market
    # loss unconditionally, so a single-tick move worse than the
    # liquidation check anticipated could debit far more than the
    # position ever had backing it.
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1000.0})
    pos = client.position("BTCUSDT")
    pos.qty = 1.0
    pos.avg_price = 10_000.0  # bought far above the fixed FakeSession price
    pos.margin_committed = 100.0  # only $100 actually backs this position
    balance_before = client.margin_balance

    await client.place_market_order("BTCUSDT", "Sell", 1.0)  # closes at bid 100.4 — a huge raw loss

    fee = (1.0 * 100.4) * (client.fee_rate_percent / 100)
    # Capped loss == exactly the committed margin, not the ~9900/unit raw move.
    assert client.margin_balance == pytest.approx(balance_before - fee)
    assert client.margin_balance > 0
    closed_pos = client.position("BTCUSDT")
    assert closed_pos.qty == pytest.approx(0.0)
    assert closed_pos.margin_committed == pytest.approx(0.0)


async def test_opening_an_order_that_needs_more_margin_than_available_raises(client):
    # Regression: this used to just do `margin_balance -= margin_required
    # + fee` unconditionally, letting the balance go negative — something
    # impossible on a real exchange, which rejects an unmargin-able
    # order. client has 1000.0 USDT; a qty of 15 at ask 100.6 (leverage 1,
    # the default) needs 1509+ in margin.
    with pytest.raises(InsufficientMarginError):
        await client.place_market_order("BTCUSDT", "Buy", 15.0)


async def test_a_rejected_order_leaves_balance_and_position_untouched(client):
    with pytest.raises(InsufficientMarginError):
        await client.place_market_order("BTCUSDT", "Buy", 15.0)
    assert client.margin_balance == pytest.approx(1000.0)
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(0.0)
    assert pos.margin_committed == pytest.approx(0.0)


async def test_leverage_brings_an_otherwise_unaffordable_order_within_reach(client):
    # The SAME order size that fails at 1x succeeds once leverage divides
    # the margin requirement down far enough — confirms the guard checks
    # actual required MARGIN, not raw notional.
    await client.ensure_leverage("BTCUSDT", 11.0)
    await client.place_market_order("BTCUSDT", "Buy", 15.0)  # notional 1509, margin ~137.2
    assert client.position("BTCUSDT").qty == pytest.approx(15.0)


async def test_a_safety_order_style_add_can_still_be_rejected_after_a_prior_fill():
    # The realistic shape of the real incident: a first fill succeeds and
    # locks some margin, then a LATER add on the same position is the one
    # that finally can't be margined — not just a single oversized order.
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 150.0})
    await client.place_market_order("BTCUSDT", "Buy", 1.0)  # notional ~100.6, affordable
    with pytest.raises(InsufficientMarginError):
        await client.place_market_order("BTCUSDT", "Buy", 1.0)  # a second ~100.6 exceeds what's left
    pos = client.position("BTCUSDT")
    assert pos.qty == pytest.approx(1.0)  # unchanged by the rejected second fill


async def test_is_open_recognizes_a_short_position(client):
    await client.place_market_order("BTCUSDT", "Sell", 1.0)
    assert client.position("BTCUSDT").is_open is True


async def test_never_calls_the_authenticated_place_order_endpoint(client):
    # PaperExchangeClient must never touch an authenticated trading
    # endpoint — orders are simulated entirely against the fake margin balance.
    session: FakeSession = client._session
    await client.place_market_order("BTCUSDT", "Buy", 1.0)
    await client.place_market_order("BTCUSDT", "Sell", 0.5)
    assert session.placed_orders == []


class _FundingRateSession(FakeSession):
    def __init__(self, funding_rate: str):
        super().__init__()
        self._funding_rate = funding_rate

    def get_tickers(self, **kwargs):
        result = super().get_tickers(**kwargs)
        result["result"]["list"][0]["fundingRate"] = self._funding_rate
        return result


async def test_get_current_funding_rate_parses_the_ticker_field():
    client = PaperExchangeClient(_FundingRateSession("0.0001"), initial_balances={"USDT": 1000.0})
    rate = await client.get_current_funding_rate("BTCUSDT")
    assert rate == pytest.approx(0.0001)


async def test_get_current_funding_rate_defaults_to_zero_when_absent(client):
    # The shared FakeSession's get_tickers has no fundingRate field —
    # must not raise, must come back as a clean 0.0.
    rate = await client.get_current_funding_rate("BTCUSDT")
    assert rate == pytest.approx(0.0)


def test_apply_funding_cost_debits_a_positive_cost(client):
    balance_before = client.margin_balance
    client.apply_funding_cost(5.0)
    assert client.margin_balance == pytest.approx(balance_before - 5.0)


def test_apply_funding_cost_credits_a_negative_cost():
    client = PaperExchangeClient(FakeSession(), initial_balances={"USDT": 1000.0})
    balance_before = client.margin_balance
    client.apply_funding_cost(-3.0)
    assert client.margin_balance == pytest.approx(balance_before + 3.0)
