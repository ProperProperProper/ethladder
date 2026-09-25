import asyncio

import pytest

import run_everything as co
from symbot_python.exchange.keychain import RealBalance
from symbot_python.strategy.optimization_store import connect as real_connect


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    path = tmp_path / "optimizer_status.json"
    monkeypatch.setattr(co, "STATUS_PATH", path)
    return path


@pytest.fixture
def fake_conn(tmp_path, monkeypatch):
    conn = real_connect(tmp_path / "test_optimization_results.db")
    monkeypatch.setattr(co, "connect_optimizer_store", lambda: conn)
    yield conn
    conn.close()


async def _fake_balance():
    return RealBalance(total_equity=1000.0, total_available_balance=1000.0, coin_balances={"USDT": 1000.0})


def test_run_cycle_isolates_one_intervals_failure(monkeypatch, status_path, fake_conn):
    # This process runs unattended for weeks — one interval hitting a
    # transient API error must not cost the other three their update.
    monkeypatch.setattr(co, "fetch_real_balance", _fake_balance)
    monkeypatch.setattr(co, "INTERVALS", ["15", "30", "60"])

    async def fake_run_interval(conn, session, symbol, interval, starting_equity, cycle_number):
        if interval == "30":
            raise RuntimeError("simulated transient API failure")
        return interval, 1.0, {"trades": 1}

    monkeypatch.setattr(co, "run_optimizer_interval", fake_run_interval)

    # Must not raise — the whole point of the fix.
    asyncio.run(co.run_optimizer_cycle(session=None, cycle_number=1))

    import json
    data = json.loads(status_path.read_text())
    # The failing interval's error must be visible, but the cycle still
    # picks an overall winner from the two intervals that succeeded.
    assert data["last_cycle_winning_interval"] in ("15", "60")


def test_run_cycle_survives_every_interval_failing(monkeypatch, status_path, fake_conn):
    monkeypatch.setattr(co, "fetch_real_balance", _fake_balance)
    monkeypatch.setattr(co, "INTERVALS", ["15", "30"])

    async def always_fails(conn, session, symbol, interval, starting_equity, cycle_number):
        raise RuntimeError("simulated total outage")

    monkeypatch.setattr(co, "run_optimizer_interval", always_fails)

    # Must not raise even when nothing succeeds this cycle.
    asyncio.run(co.run_optimizer_cycle(session=None, cycle_number=1))


class _HangingRiskLimitSession:
    def get_risk_limit(self, **kwargs):
        import time
        # Long enough to guarantee it outlasts the 0.05s timeout below
        # (proving wait_for actually gives up rather than blocking), short
        # enough that asyncio.run()'s shutdown_default_executor() — which
        # waits for in-flight to_thread() threads to finish, since
        # wait_for's cancellation only gives up on AWAITING, it can't kill
        # the underlying OS thread — doesn't hold up this test for long.
        time.sleep(2.0)

    def get_kline(self, **kwargs):
        raise AssertionError("should not be called in this test")


def test_fetch_fresh_market_data_risk_limit_call_is_timeout_bounded(monkeypatch):
    # Real bug class this guards against: continuous_optimizer.py makes
    # its own direct pybit calls (not through the already-timeout-wrapped
    # exchange client classes), and previously had zero timeout on any of
    # them — a single stalled Bybit call would hang the whole optimizer
    # process forever with no error logged.
    async def fast_timeout(awaitable, timeout=0.05):
        return await asyncio.wait_for(awaitable, timeout=0.05)

    monkeypatch.setattr(co, "call_with_timeout", fast_timeout)

    async def fake_fetch_klines(symbol, interval, limit):
        return [[0, 100, 101, 99, 100, 1], [1, 100, 101, 99, 100, 1]]

    monkeypatch.setattr(co, "fetch_klines", fake_fetch_klines)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await co.fetch_fresh_market_data(_HangingRiskLimitSession(), "ETHUSDT", "15")

    asyncio.run(run())


def test_partial_walk_forward_cannot_be_promoted(monkeypatch, status_path, fake_conn):
    from types import SimpleNamespace
    from symbot_python.strategy.optimize import WalkForwardResult

    candles = [[i * 3_600_000, 100, 100, 100, 100, 1] for i in range(14 * 24)]

    async def market(*args):
        return candles, [], []

    session = SimpleNamespace(get_instruments_info=lambda **kw: {
        "result": {"list": [{"priceFilter": {"tickSize": "0.01"},
                                "lotSizeFilter": {"qtyStep": "0.001"}}]}})
    monkeypatch.setattr(co, "fetch_fresh_market_data", market)
    monkeypatch.setattr(co, "walk_forward", lambda *a, **kw: WalkForwardResult(
        windows=[object()], combined_out_of_sample_trades=[],
        combined_final_equity=1100, starting_equity=1000))

    def unexpected(*args, **kwargs):
        pytest.fail("incomplete evaluation must not be recorded or promoted")

    monkeypatch.setattr(co, "record_optimization", unexpected)
    monkeypatch.setattr(co, "promote_to_winner", unexpected)
    asyncio.run(co.run_optimizer_interval(fake_conn, session, "ETHUSDT", "60", 1000, 1))

    import json
    status = json.loads(status_path.read_text())
    assert status["windows_total"] == 3
