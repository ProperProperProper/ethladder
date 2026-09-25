import pytest

from symbot_python.api import app as app_module
from symbot_python.signals.candles import resample_candles


class StubSession:
    """Simulates Bybit's get_kline pagination contract (respects `end`
    and `limit`, returns newest-first) over a synthetic 1-minute series,
    deriving any requested native interval by aggregating that series —
    good enough to exercise pagination + custom-interval logic without
    a real network call.
    """

    def __init__(self, total_minutes: int, start_ts_ms: int = 0):
        self.total_minutes = total_minutes
        self.start_ts_ms = start_ts_ms
        self.calls: list[dict] = []

    def _base_1m_rows(self):
        return [
            [
                self.start_ts_ms + i * 60_000,
                100.0 + i * 0.01,
                100.5 + i * 0.01,
                99.5 + i * 0.01,
                100.2 + i * 0.01,
                1.0,
            ]
            for i in range(self.total_minutes)
        ]

    def get_kline(self, **kwargs):
        self.calls.append(dict(kwargs))
        interval = kwargs["interval"]
        limit = kwargs["limit"]
        end = kwargs.get("end")

        rows = self._base_1m_rows()
        if interval != "1":
            rows = resample_candles(rows, int(interval))
        if end is not None:
            rows = [r for r in rows if r[0] <= end]
        page = rows[-limit:] if len(rows) > limit else rows
        bybit_rows = [
            [str(int(r[0])), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), "0"]
            for r in reversed(page)  # newest-first, matching Bybit
        ]
        return {"retCode": 0, "retMsg": "OK", "result": {"list": bybit_rows}}


@pytest.fixture(autouse=True)
def restore_session(monkeypatch):
    yield
    # nothing to restore explicitly; each test sets its own stub


async def test_single_page_native_interval(monkeypatch):
    stub = StubSession(total_minutes=2000)
    monkeypatch.setattr(app_module, "_public_session", stub)
    candles = await app_module.fetch_klines("BTCUSDT", "60", 10)
    assert len(candles) == 10
    # oldest-first
    assert candles[0][0] < candles[-1][0]
    assert len(stub.calls) == 1


async def test_native_interval_paginates_beyond_1000(monkeypatch):
    # 2500 one-minute candles = far more than one page's worth
    stub = StubSession(total_minutes=3000)
    monkeypatch.setattr(app_module, "_public_session", stub)
    candles = await app_module.fetch_klines("BTCUSDT", "1", 2500)
    assert len(candles) == 2500
    # strictly increasing, contiguous 1-minute timestamps, no gaps/dupes
    diffs = {candles[i + 1][0] - candles[i][0] for i in range(len(candles) - 1)}
    assert diffs == {60_000.0}
    assert len(stub.calls) >= 3  # required more than one page (1000/page)


async def test_custom_interval_29m_aggregates_from_1m(monkeypatch):
    stub = StubSession(total_minutes=5000)
    monkeypatch.setattr(app_module, "_public_session", stub)
    candles = await app_module.fetch_klines("BTCUSDT", "29", 50)
    assert len(candles) == 50
    diffs = {candles[i + 1][0] - candles[i][0] for i in range(len(candles) - 1)}
    assert diffs == {29 * 60_000.0}
    # confirms it actually paginated the underlying 1-minute fetch
    assert all(call["interval"] == "1" for call in stub.calls)


async def test_custom_interval_past_14_days(monkeypatch):
    # 14 days of 1-minute data = 20160 minutes; request enough 59m bars
    # to span it and confirm we actually get that much real time covered.
    total_minutes = 21000
    stub = StubSession(total_minutes=total_minutes)
    monkeypatch.setattr(app_module, "_public_session", stub)
    limit = (14 * 24 * 60) // 59 + 5  # comfortably past 14 days of 59m bars
    candles = await app_module.fetch_klines("BTCUSDT", "59", limit)
    span_minutes = (candles[-1][0] - candles[0][0]) / 60_000
    assert span_minutes >= 14 * 24 * 60


async def test_runs_out_of_history_returns_partial_without_hanging(monkeypatch):
    stub = StubSession(total_minutes=50)
    monkeypatch.setattr(app_module, "_public_session", stub)
    candles = await app_module.fetch_klines("BTCUSDT", "1", 500)
    assert len(candles) == 50  # can't return more than exists
