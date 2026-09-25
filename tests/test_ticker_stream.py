import time

import symbot_python.exchange.ticker_stream as ticker_stream_module
from symbot_python.exchange.ticker_stream import STALE_AFTER_SEC, TickerStream


class _FakeWebSocket:
    """Stands in for pybit's WebSocket — records subscriptions instead of
    opening a real connection, so these tests never touch the network."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.subscriptions: list[tuple[str, object]] = []
        self.exited = False
        _FakeWebSocket.instances.append(self)

    def ticker_stream(self, symbol, callback):
        self.subscriptions.append((symbol, callback))

    def exit(self):
        self.exited = True


def _sample_message(symbol="ETHUSDT", last_price="2500.5"):
    return {
        "topic": f"tickers.{symbol}",
        "type": "snapshot",
        "data": {
            "symbol": symbol,
            "lastPrice": last_price,
            "bid1Price": "2500.0",
            "ask1Price": "2501.0",
            "volume24h": "1000",
            "turnover24h": "2500000",
        },
    }


def test_get_price_returns_none_before_any_message():
    stream = TickerStream()
    assert stream.get_price("ETHUSDT") is None


def test_on_message_caches_the_latest_price():
    stream = TickerStream()
    stream._on_message(_sample_message())
    ticker = stream.get_price("ETHUSDT")
    assert ticker is not None
    assert ticker.symbol == "ETHUSDT"
    assert ticker.last == 2500.5
    assert ticker.bid == 2500.0
    assert ticker.ask == 2501.0


def test_on_message_ignores_a_malformed_message_without_raising():
    stream = TickerStream()
    stream._on_message({"data": {"symbol": "ETHUSDT"}})  # no lastPrice
    assert stream.get_price("ETHUSDT") is None
    stream._on_message({})  # no data at all
    assert stream.get_price("ETHUSDT") is None


def test_get_price_returns_none_once_the_cached_price_goes_stale(monkeypatch):
    stream = TickerStream()
    stream._on_message(_sample_message())
    assert stream.get_price("ETHUSDT") is not None

    real_monotonic = time.monotonic()
    monkeypatch.setattr(
        ticker_stream_module.time, "monotonic", lambda: real_monotonic + STALE_AFTER_SEC + 1
    )
    assert stream.get_price("ETHUSDT") is None


def test_ensure_subscribed_only_subscribes_once_per_symbol(monkeypatch):
    _FakeWebSocket.instances.clear()
    monkeypatch.setattr(ticker_stream_module, "WebSocket", _FakeWebSocket)

    stream = TickerStream()
    stream.ensure_subscribed("ETHUSDT")
    stream.ensure_subscribed("ETHUSDT")
    stream.ensure_subscribed("ETHUSDT")

    assert len(_FakeWebSocket.instances) == 1
    assert len(_FakeWebSocket.instances[0].subscriptions) == 1
    assert _FakeWebSocket.instances[0].subscriptions[0][0] == "ETHUSDT"


def test_ensure_subscribed_reuses_one_websocket_for_a_second_symbol(monkeypatch):
    _FakeWebSocket.instances.clear()
    monkeypatch.setattr(ticker_stream_module, "WebSocket", _FakeWebSocket)

    stream = TickerStream()
    stream.ensure_subscribed("ETHUSDT")
    stream.ensure_subscribed("BTCUSDT")

    assert len(_FakeWebSocket.instances) == 1  # one shared connection
    assert len(_FakeWebSocket.instances[0].subscriptions) == 2


def test_stop_closes_the_websocket_and_clears_the_cache(monkeypatch):
    _FakeWebSocket.instances.clear()
    monkeypatch.setattr(ticker_stream_module, "WebSocket", _FakeWebSocket)

    stream = TickerStream()
    stream.ensure_subscribed("ETHUSDT")
    stream._on_message(_sample_message())
    assert stream.get_price("ETHUSDT") is not None

    stream.stop()

    assert _FakeWebSocket.instances[0].exited is True
    assert stream.get_price("ETHUSDT") is None


def test_stop_is_safe_to_call_when_never_subscribed():
    stream = TickerStream()
    stream.stop()  # must not raise


def test_ensure_subscribed_constructs_a_real_pybit_websocket_without_error():
    """Regression: WebSocket(channel_type=..., retries=0) alone raised
    'missing 1 required positional argument: testnet' at construction
    time — a real pybit.unified_trading.WebSocket has no default for it.
    The other tests all monkeypatch WebSocket away and can't catch this
    class of bug by design, so this one constructs the REAL class (cheap
    and network-free: pybit's __init__ only sets attributes, the actual
    connection is opened lazily by .ticker_stream()/.subscribe(), never
    called here) with the exact kwargs ensure_subscribed passes.
    """
    from pybit.unified_trading import WebSocket

    ws = WebSocket(channel_type=ticker_stream_module.CATEGORY, testnet=False, retries=0)
    assert ws is not None
