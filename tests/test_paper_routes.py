import re
import time

import pytest
from fastapi.testclient import TestClient

from symbot_python.api import app as app_module
from symbot_python.api import paper as paper_module
from symbot_python.exchange.keychain import RealBalance
from symbot_python.strategy.backtest import MIN_WIN_PRICE_PCT
from symbot_python.exchange.paper_client import PaperExchangeClient
from symbot_python.strategy import dca_bot
from symbot_python.strategy.models import DealStatus
from tests.test_bybit_client import FakeSession


class _FakeConn:
    """Stands in for optimization_store.connect()'s sqlite3.Connection so
    these tests never touch the real, on-disk param library (which the
    continuous optimizer may be actively writing to in the background).
    """

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    monkeypatch.setattr(dca_bot, "TICK_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(dca_bot, "RETRY_INTERVAL_SEC", 0.005)


def stub_factory(session_cls=FakeSession, balances=None):
    async def fake_create_exchange_client(mode, **kwargs):
        return PaperExchangeClient(session_cls(), initial_balances=balances or {"USDT": 10_000.0})

    return fake_create_exchange_client


async def _fake_fetch_real_balance():
    return RealBalance(total_equity=10_000.0, total_available_balance=10_000.0, coin_balances={"USDT": 10_000.0})


@pytest.fixture
def client(monkeypatch):
    # Route the lazy manager construction (still exercised for real —
    # including its real `.start()` call inside the app's own event loop)
    # through a stub session instead of live Bybit, so these tests are
    # offline and deterministic without bypassing any app wiring, without
    # ever touching the real Keychain/account balance, and without ever
    # reading the real (possibly concurrently-written) param library DB.
    monkeypatch.setattr(paper_module, "_manager", None)
    monkeypatch.setattr(paper_module, "_refresh_task", None)
    monkeypatch.setattr(paper_module, "_last_param_sync", None)
    monkeypatch.setattr(paper_module, "create_exchange_client", stub_factory())
    monkeypatch.setattr(paper_module, "fetch_real_balance", _fake_fetch_real_balance)
    monkeypatch.setattr(paper_module, "connect", lambda: _FakeConn())
    monkeypatch.setattr(paper_module, "get_best_current_winner", lambda conn, symbol: None)
    with TestClient(app_module.app) as c:
        yield c


def wait_until(condition, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def test_paper_page_builds_manager_and_auto_bot(client):
    response = client.get("/paper")
    assert response.status_code == 200
    assert paper_module._manager is not None
    # Not an exact-equality check: the auto bot's deal can start filling
    # (and, since paper_client.py now simulates funding, accrue a small
    # real cost) in the background between get_manager() returning and
    # this assertion running — an exact 10_000.0 here was flaky under
    # load (observed failing ~1/3 runs under heavy concurrent CPU
    # contention). The real thing being verified is "seeded from a real
    # balance, not a fake round number, and hasn't gone haywire" — a
    # generous band still catches that without racing the tick loop.
    assert 9_000.0 < paper_module._manager.exchange.margin_balance <= 10_000.0
    # No winner recorded (mocked to None) -> the auto bot still gets
    # created from fallback defaults, not left absent.
    wait_until(lambda: len(paper_module._manager.bots) == 1)
    bot = next(iter(paper_module._manager.bots.values()))
    assert bot.pair == "ETH/USDT"
    assert bot.leverage == pytest.approx(11.0)
    assert paper_module._last_param_sync["used_fallback"] is True


def test_external_telemetry_export_is_removed(client):
    assert client.get("/api/telemetry").status_code == 404


def test_local_dashboard_status_contains_only_trade_counts(client):
    body = client.get("/paper/status").json()
    assert set(body) == {"total_trades", "open_deal_count"}
    assert body["total_trades"] == 0
    page = client.get("/paper").text
    assert "fetch('/paper/status')" in page
    assert "/api/telemetry" not in page


def test_auto_bot_starts_a_deal(client):
    client.get("/paper")
    manager = paper_module._manager
    wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    wait_until(lambda: deal.is_start == 1)
    assert deal.filled_count >= 1


def test_update_params_keeps_take_profit_fixed_when_applying_a_winner(client, monkeypatch):
    client.get("/paper")  # creates the auto bot from fallback defaults first
    manager = paper_module._manager
    bot = next(iter(manager.bots.values()))
    assert bot.dca_take_profit_percent != 2.5  # sanity: not already this value

    def fake_winner(conn, symbol):
        return {
            "interval": "60", "in_sample_score": 42.0, "tested_at": "2026-01-01T00:00:00Z",
            "params_json": None,
        }

    # row_params() normally parses params_json off a real sqlite3.Row;
    # patch it directly so this test doesn't need a real DB row shape.
    monkeypatch.setattr(paper_module, "get_best_current_winner", fake_winner)
    monkeypatch.setattr(
        paper_module, "row_params",
        lambda row: {"dca_take_profit_percent": 2.5, "side": "short", "dca_max_order": 6},
    )

    response = client.post("/paper/update-params", follow_redirects=False)
    assert response.status_code == 303
    # Same bot instance, mutated in place — not a second bot created.
    assert len(manager.bots) == 1
    assert bot.dca_take_profit_percent == pytest.approx(0.33)
    assert bot.side == "short"
    assert bot.dca_max_order == 6
    assert bot.leverage == pytest.approx(11.0)  # winner_params has no "leverage" key -> DEFAULT_LEVERAGE


def test_update_params_clamps_a_sub_floor_take_profit_from_a_stale_winner_row(client, monkeypatch):
    # Non-negotiable floor (see backtest.clamp_take_profit_percent): a
    # param-library row recorded before this floor existed (or ever
    # manually set below it) must never reach the live/paper bot with a
    # take-profit that can never register as a real win by construction.
    client.get("/paper")
    manager = paper_module._manager
    bot = next(iter(manager.bots.values()))

    def fake_winner(conn, symbol):
        return {"interval": "60", "in_sample_score": 1.0, "tested_at": "2026-01-01T00:00:00Z", "params_json": None}

    monkeypatch.setattr(paper_module, "get_best_current_winner", fake_winner)
    monkeypatch.setattr(paper_module, "row_params", lambda row: {"dca_take_profit_percent": 0.05})

    response = client.post("/paper/update-params", follow_redirects=False)
    assert response.status_code == 303
    assert bot.dca_take_profit_percent == pytest.approx(MIN_WIN_PRICE_PCT)
    assert paper_module._last_param_sync["used_fallback"] is False
    assert paper_module._last_param_sync["source_interval"] == "60"


def test_update_params_adopts_a_winner_leverage_within_the_band(client, monkeypatch):
    client.get("/paper")
    manager = paper_module._manager
    bot = next(iter(manager.bots.values()))

    monkeypatch.setattr(
        paper_module, "get_best_current_winner",
        lambda conn, symbol: {"interval": "60", "in_sample_score": 42.0, "tested_at": "t", "params_json": None},
    )
    monkeypatch.setattr(paper_module, "row_params", lambda row: {"leverage": 9.0})

    client.post("/paper/update-params", follow_redirects=False)
    assert bot.leverage == pytest.approx(9.0)  # within [9,11] -> adopted as-is, not forced to 11


@pytest.mark.parametrize("winner_leverage,expected_clamped", [(20.0, 11.0), (1.0, 9.0), (11.0000001, 11.0)])
def test_update_params_clamps_an_out_of_band_winner_leverage(client, monkeypatch, winner_leverage, expected_clamped):
    # "leverage should never exceed 11x and never lower than 9x" — even a
    # stray/corrupted param-library row must come out clamped, never
    # applied to the live bot as-is.
    client.get("/paper")
    manager = paper_module._manager
    bot = next(iter(manager.bots.values()))

    monkeypatch.setattr(
        paper_module, "get_best_current_winner",
        lambda conn, symbol: {"interval": "60", "in_sample_score": 42.0, "tested_at": "t", "params_json": None},
    )
    monkeypatch.setattr(paper_module, "row_params", lambda row: {"leverage": winner_leverage})

    client.post("/paper/update-params", follow_redirects=False)
    assert bot.leverage == pytest.approx(expected_clamped)


@pytest.mark.parametrize("winner_utilization,expected_clamped", [(10.0, 25.0), (0.0, 25.0), (150.0, 98.0), (60.0, 60.0)])
def test_update_params_clamps_an_out_of_band_winner_funds_utilization(client, monkeypatch, winner_utilization, expected_clamped):
    # "positions should not be lower than 25% first entry" — a deal sized
    # below the floor commits too little of the account for even a
    # fully-successful ladder to produce a meaningful profit. Same
    # defense-in-depth as leverage: a stray/corrupted or pre-floor
    # param-library row must come out clamped, never applied as-is.
    client.get("/paper")
    manager = paper_module._manager
    bot = next(iter(manager.bots.values()))

    monkeypatch.setattr(
        paper_module, "get_best_current_winner",
        lambda conn, symbol: {"interval": "60", "in_sample_score": 42.0, "tested_at": "t", "params_json": None},
    )
    monkeypatch.setattr(paper_module, "row_params", lambda row: {"funds_utilization_percent": winner_utilization})

    client.post("/paper/update-params", follow_redirects=False)
    assert bot.funds_utilization_percent == pytest.approx(expected_clamped)


class _FlatPriceFakeSession(FakeSession):
    """Every price (last/bid/ask) identical, so a filled deal's average
    exactly equals the current price — the break-even (0.0%) edge case.
    """

    def get_tickers(self, **kwargs):
        return {
            "retCode": 0, "retMsg": "OK",
            "result": {"list": [{
                "lastPrice": "100.0", "bid1Price": "100.0", "ask1Price": "100.0",
                "volume24h": "1", "turnover24h": "1",
            }]},
        }


def test_paper_page_shows_breakeven_deal_as_not_negative(client, monkeypatch):
    # Regression: the template used `d.unrealized_percent and ... >= 0` to
    # pick the CSS class — since 0.0 is falsy in Jinja2/Python, a deal
    # sitting at exactly break-even was miscolored "negative" instead of
    # neutral/positive. Fixed to `is not none`.
    monkeypatch.setattr(paper_module, "create_exchange_client", stub_factory(_FlatPriceFakeSession))
    response = client.get("/paper")
    wait_until(lambda: len(paper_module._manager.deals) == 1)
    deal = next(iter(paper_module._manager.deals.values()))
    wait_until(lambda: deal.is_start == 1)

    response = client.get("/paper")
    assert response.status_code == 200
    # The deal's unrealized percent cell must render "positive", never
    # "negative", when it is sitting at exactly 0.00%.
    assert re.search(r'<td class="positive"[^>]*>\s*0\.00%', response.text)
    assert not re.search(r'<td class="negative"[^>]*>\s*0\.00%', response.text)


def test_clear_circuit_breaker_route(client):
    client.get("/paper")
    manager = paper_module._manager
    manager.circuit_breaker_active = True

    response = client.post("/paper/clear-circuit-breaker", follow_redirects=False)
    assert response.status_code == 303
    assert manager.circuit_breaker_active is False


def test_stop_bot_disables_it(client):
    client.get("/paper")
    manager = paper_module._manager
    wait_until(lambda: len(manager.bots) == 1)
    bot = next(iter(manager.bots.values()))

    response = client.post(f"/paper/bots/{bot.bot_id}/stop", follow_redirects=False)
    assert response.status_code == 303
    assert bot.active is False


def test_cancel_deal_closes_it(client, monkeypatch):
    monkeypatch.setattr(
        paper_module, "row_params", lambda row: {"dca_take_profit_percent": 1000.0}
    )
    monkeypatch.setattr(
        paper_module, "get_best_current_winner",
        lambda conn, symbol: {"interval": "60", "in_sample_score": 1.0, "tested_at": "t"},
    )
    client.get("/paper")
    manager = paper_module._manager
    wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    wait_until(lambda: deal.is_start == 1)

    response = client.post(f"/paper/deals/{deal.deal_id}/cancel", follow_redirects=False)
    assert response.status_code == 303
    wait_until(lambda: deal.status == DealStatus.CLOSED)
    assert deal.canceled is True


def test_panic_sell_deal_closes_it(client, monkeypatch):
    monkeypatch.setattr(
        paper_module, "row_params", lambda row: {"dca_take_profit_percent": 1000.0}
    )
    monkeypatch.setattr(
        paper_module, "get_best_current_winner",
        lambda conn, symbol: {"interval": "60", "in_sample_score": 1.0, "tested_at": "t"},
    )
    client.get("/paper")
    manager = paper_module._manager
    wait_until(lambda: len(manager.deals) == 1)
    deal = next(iter(manager.deals.values()))
    wait_until(lambda: deal.is_start == 1)

    response = client.post(f"/paper/deals/{deal.deal_id}/panic", follow_redirects=False)
    assert response.status_code == 303
    wait_until(lambda: deal.status == DealStatus.CLOSED)
    assert deal.panic_sell is True


def test_optimizer_direction_change_updates_future_deals_without_closing_active_deal(client, monkeypatch):
    client.get("/paper")
    manager = paper_module._manager
    wait_until(lambda: any(d.is_start for d in manager.deals.values()))
    original = next(iter(manager.deals.values()))
    monkeypatch.setattr(paper_module, "_params_from_best_winner", lambda: {
        "config": {**paper_module.FALLBACK_BOT_DEFAULTS, "side": "short"},
        "source_interval": "60", "source_score": 42,
        "source_tested_at": "2026-09-18T00:00:00Z",
    })
    client.post("/paper/update-params")
    assert original.status == DealStatus.ACTIVE
    assert original.sell_data is None
    assert original.config.side == "long"
    bot = next(b for b in manager.bots.values() if b.bot_name == paper_module.AUTO_BOT_NAME)
    assert bot.side == "short"
    page = client.get("/paper").text
    assert "Optimizer settings:" in page


def test_equity_chart_visible_before_any_trade_closes(client):
    page = client.get('/paper').text
    assert 'id="equity-chart"' in page
    assert 'Account equity' in page
    chart = client.get('/paper/equity')
    assert chart.status_code == 200
    assert '<svg' in chart.text
    assert 'USDT' in chart.text
    assert client.get('/api/telemetry').status_code == 404
