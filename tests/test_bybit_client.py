import pytest

from symbot_python.exchange.base import TradingMode
from symbot_python.exchange.bybit_client import BybitApiError, BybitClient


class FakeSession:
    """Stub implementing the BybitSession surface with canned Bybit-shaped
    responses, so these tests never touch the network.
    """

    def __init__(self):
        self.placed_orders = []
        self.cancelled = []
        self._order_status = "New"

    def get_wallet_balance(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": [{"coin": [{"coin": "USDT", "walletBalance": "1000.5"}]}]},
        }

    def get_tickers(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "lastPrice": "100.5",
                        "bid1Price": "100.4",
                        "ask1Price": "100.6",
                        "volume24h": "12345",
                        "turnover24h": "1234500",
                    }
                ]
            },
        }

    def get_instruments_info(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {"basePrecision": "0.0001", "minOrderQty": "0.0001", "minOrderAmt": "5"},
                        "priceFilter": {"tickSize": "0.01"},
                    }
                ]
            },
        }

    def place_order(self, **kwargs):
        self.placed_orders.append(kwargs)
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "abc123"}}

    def get_order_history(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "orderStatus": "Filled",
                        "avgPrice": "100.5",
                        "cumExecQty": "1.0",
                        "cumExecValue": "100.5",
                        "cumExecFee": "0.09",
                    }
                ]
            },
        }

    def get_open_orders(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "orderStatus": self._order_status,
                        "avgPrice": "0",
                        "cumExecQty": "0",
                        "cumExecValue": "0",
                        "cumExecFee": "0",
                    }
                ]
                if self._order_status != "gone"
                else []
            },
        }

    def cancel_order(self, **kwargs):
        self.cancelled.append(kwargs)
        return {"retCode": 0, "retMsg": "OK", "result": {}}

    def get_kline(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    ["3000", "101", "102", "99", "100.5", "10", "1000"],
                    ["2000", "100", "103", "98", "101", "12", "1200"],
                    ["1000", "99", "101", "97", "100", "8", "800"],
                ]
            },
        }

    def set_leverage(self, **kwargs):
        self.leverage_calls = getattr(self, "leverage_calls", [])
        self.leverage_calls.append(kwargs)
        return {"retCode": 0, "retMsg": "OK", "result": {}}

    def get_positions(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "symbol": kwargs.get("symbol"), "side": "Buy", "size": "1.5",
                        "avgPrice": "100.0", "leverage": "11", "liqPrice": "91.4",
                        "unrealisedPnl": "5.25",
                    }
                ]
            },
        }


class FlatPositionSession(FakeSession):
    def get_positions(self, **kwargs):
        return {"retCode": 0, "retMsg": "OK", "result": {"list": [{"size": "0"}]}}


class LeverageAlreadySetSession(FakeSession):
    def set_leverage(self, **kwargs):
        return {"retCode": 110043, "retMsg": "leverage not modified", "result": {}}


class FakeErrorSession(FakeSession):
    def get_wallet_balance(self, **kwargs):
        return {"retCode": 10001, "retMsg": "invalid api key", "result": {}}


@pytest.fixture
def client():
    return BybitClient(FakeSession())


def test_mode_is_live():
    assert BybitClient(FakeSession()).mode == TradingMode.LIVE


async def test_verify_connection_success(client):
    await client.verify_connection()  # should not raise


async def test_verify_connection_raises_on_api_error():
    client = BybitClient(FakeErrorSession())
    with pytest.raises(BybitApiError):
        await client.verify_connection()


async def test_get_balance_parses_coins(client):
    balances = await client.get_balance()
    assert balances["USDT"] == pytest.approx(1000.5)


async def test_get_precision_and_filters(client):
    precision = await client.get_precision("BTCUSDT")
    assert precision.tick_size == pytest.approx(0.01)
    assert precision.qty_step == pytest.approx(0.0001)
    assert client.filter_price(precision, 100.567) == pytest.approx(100.56)
    assert client.filter_amount(precision, 1.00009) == pytest.approx(1.0)


async def test_get_precision_is_cached(client):
    first = await client.get_precision("BTCUSDT")
    second = await client.get_precision("BTCUSDT")
    assert first is second


async def test_get_ticker(client):
    ticker = await client.get_ticker("BTCUSDT")
    assert ticker.last == pytest.approx(100.5)
    assert ticker.bid == pytest.approx(100.4)
    assert ticker.ask == pytest.approx(100.6)


async def test_get_kline_reversed_to_oldest_first(client):
    candles = await client.get_kline("BTCUSDT", "5")
    # Bybit returns newest-first; wrapper must reverse to oldest-first.
    assert candles[0][0] == 1000.0
    assert candles[-1][0] == 3000.0


async def test_place_market_order(client):
    session = client._session
    result = await client.place_market_order("BTCUSDT", "Buy", 1.0)
    assert result.order_id == "abc123"
    assert session.placed_orders[0]["orderType"] == "Market"
    assert "price" not in session.placed_orders[0]


async def test_place_market_order_missing_order_id_raises():
    class NoOrderIdSession(FakeSession):
        def place_order(self, **kwargs):
            return {"retCode": 0, "retMsg": "OK", "result": {}}

    client = BybitClient(NoOrderIdSession())
    with pytest.raises(BybitApiError):
        await client.place_market_order("BTCUSDT", "Buy", 1.0)


async def test_get_order_status_filled_from_history_when_not_open(client):
    client._session._order_status = "gone"
    status = await client.get_order_status("BTCUSDT", "abc123")
    assert status.status == "filled"
    assert status.avg_price == pytest.approx(100.5)


async def test_get_order_status_open(client):
    client._session._order_status = "New"
    status = await client.get_order_status("BTCUSDT", "abc123")
    assert status.status == "open"


async def test_verify_order_returns_immediately_on_fill():
    session = FakeSession()
    session._order_status = "gone"  # forces the history (filled) path
    client = BybitClient(session)
    status = await client.verify_order("BTCUSDT", "abc123")
    assert status.status == "filled"


async def test_default_category_is_linear(client):
    assert client.category == "linear"


async def test_buy_order_is_not_reduce_only_when_adding_to_an_existing_long(client):
    # default FakeSession reports an existing "Buy" (long) position
    await client.place_market_order("BTCUSDT", "Buy", 1.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is False
    assert client._session.placed_orders[-1]["positionIdx"] == 0


async def test_sell_order_is_reduce_only_when_closing_an_existing_long(client):
    await client.place_market_order("BTCUSDT", "Sell", 1.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is True


async def test_sell_order_is_still_reduce_only_when_exactly_closing_the_position(client):
    # default FakeSession position size is 1.5 — exactly that qty stays
    # a plain, safe reduce.
    await client.place_market_order("BTCUSDT", "Sell", 1.5)
    assert client._session.placed_orders[-1]["reduceOnly"] is True


async def test_sell_order_is_not_reduce_only_when_sized_to_close_and_flip(client):
    # Oversized relative to the existing 1.5 long — a deliberate
    # close-and-reverse (see dca_bot.py's reverse_drawdown handling),
    # not a plain reduce. Regression: reduceOnly=True here would have
    # Bybit reject or clamp the order, so a stop-and-reverse built by
    # simply sending a bigger qty would never actually flip on a real
    # account, even though the exact same order works correctly against
    # PaperExchangeClient's equivalent flip branch in simulation.
    await client.place_market_order("BTCUSDT", "Sell", 2.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is False


async def test_sell_order_opens_a_short_and_is_not_reduce_only_when_flat():
    # Previously reduceOnly was hardcoded from side alone (side=="Sell"),
    # which would have made Bybit reject a short strategy's own opening
    # order outright — reduceOnly requires an existing opposite position
    # to reduce, and there is none when opening from flat.
    client = BybitClient(FlatPositionSession())
    await client.place_market_order("BTCUSDT", "Sell", 1.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is False


async def test_sell_order_adds_to_an_existing_short_and_is_not_reduce_only():
    class ShortPositionSession(FakeSession):
        def get_positions(self, **kwargs):
            return {
                "retCode": 0, "retMsg": "OK",
                "result": {"list": [{
                    "symbol": kwargs.get("symbol"), "side": "Sell", "size": "1.5",
                    "avgPrice": "100.0", "leverage": "11", "liqPrice": "108.6",
                    "unrealisedPnl": "5.25",
                }]},
            }

    client = BybitClient(ShortPositionSession())
    await client.place_market_order("BTCUSDT", "Sell", 1.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is False


async def test_buy_order_is_reduce_only_when_closing_an_existing_short():
    class ShortPositionSession(FakeSession):
        def get_positions(self, **kwargs):
            return {
                "retCode": 0, "retMsg": "OK",
                "result": {"list": [{
                    "symbol": kwargs.get("symbol"), "side": "Sell", "size": "1.5",
                    "avgPrice": "100.0", "leverage": "11", "liqPrice": "108.6",
                    "unrealisedPnl": "5.25",
                }]},
            }

    client = BybitClient(ShortPositionSession())
    await client.place_market_order("BTCUSDT", "Buy", 1.0)
    assert client._session.placed_orders[-1]["reduceOnly"] is True


async def test_ensure_leverage_calls_set_leverage_once_per_symbol(client):
    await client.ensure_leverage("BTCUSDT", 11.0)
    await client.ensure_leverage("BTCUSDT", 11.0)  # should be a no-op the second time
    assert len(client._session.leverage_calls) == 1
    assert client._session.leverage_calls[0]["buyLeverage"] == "11.0"


async def test_ensure_leverage_recalls_when_leverage_changes(client):
    await client.ensure_leverage("BTCUSDT", 11.0)
    await client.ensure_leverage("BTCUSDT", 5.0)
    assert len(client._session.leverage_calls) == 2


async def test_ensure_leverage_tolerates_already_set_error():
    client = BybitClient(LeverageAlreadySetSession())
    await client.ensure_leverage("BTCUSDT", 11.0)  # must not raise


async def test_get_position_parses_liquidation_price(client):
    position = await client.get_position("BTCUSDT")
    assert position is not None
    assert position.side == "Buy"
    assert position.size == pytest.approx(1.5)
    assert position.liquidation_price == pytest.approx(91.4)
    assert position.leverage == pytest.approx(11.0)


async def test_get_position_returns_none_when_flat():
    client = BybitClient(FlatPositionSession())
    position = await client.get_position("BTCUSDT")
    assert position is None
