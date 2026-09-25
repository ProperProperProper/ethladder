from types import SimpleNamespace

import pytest

from symbot_python.api.equity import EquityHistory
from symbot_python.api.paper import _sample_equity
from symbot_python.strategy.models import DealStatus


@pytest.mark.parametrize('qty,price,expected', [(2, 110, 118), (-2, 90, 118), (-2, 110, 78)])
async def test_equity_uses_actual_position_and_accrued_funding(qty, price, expected):
    async def ticker(symbol):
        return SimpleNamespace(last=price)
    exchange = SimpleNamespace(
        margin_balance=80, get_ticker=ticker,
        position=lambda symbol: SimpleNamespace(qty=qty, avg_price=100, margin_committed=20),
    )
    manager = SimpleNamespace(exchange=exchange, deals={
        'active': SimpleNamespace(status=DealStatus.ACTIVE, funding_cost_quote=2),
        'closed': SimpleNamespace(status=DealStatus.CLOSED, funding_cost_quote=5),
    })
    history = EquityHistory(0, 100)
    await _sample_equity(manager, history)
    assert history.points[-1][1] == expected


def test_chart_renders_at_session_start_and_bounds_retention():
    history = EquityHistory(0, 100)
    assert '<svg' in history.render()
    assert '100.00 USDT' in history.render()
    for i in range(9000):
        history.record(i + 1, 100)
    assert len(history.points) == 1440
    assert 'nan' not in history.render()
    history.record(10000, float('nan'))
    assert history.points[-1][0] == 9000


def test_drawdown_tracks_recovery_and_preserves_session_maximum():
    history = EquityHistory(0, 100)
    history.record(60, 120, realised=10, unrealised=10)
    history.record(120, 90, realised=10, unrealised=-20)
    assert history.points[-1].drawdown == 30
    assert history.points[-1].drawdown_percent == 25
    history.record(180, 110, realised=5, unrealised=5)
    assert history.points[-1].drawdown == 10
    assert history.max_drawdown == 30
    assert history.max_drawdown_percent == 25
    for i in range(1500):
        history.record(240 + i * 60, 130, realised=30, unrealised=0)
    assert history.points[-1].drawdown == 0
    assert history.peak == 130
    assert history.max_drawdown == 30
    assert history.max_drawdown_percent == 25
    assert history.points[0].timestamp > 120


async def test_realised_unrealised_reconcile_as_position_closes():
    position = SimpleNamespace(qty=2, avg_price=100, margin_committed=20)
    async def ticker(symbol):
        return SimpleNamespace(last=110)
    exchange = SimpleNamespace(margin_balance=79, get_ticker=ticker,
                               position=lambda symbol: position)
    deal = SimpleNamespace(status=DealStatus.ACTIVE, funding_cost_quote=2)
    manager = SimpleNamespace(exchange=exchange, deals={'trade': deal})
    history = EquityHistory(0, 100)
    await _sample_equity(manager, history)
    sample = history.points[-1]
    assert sample.realised == -1  # Entry fee already paid.
    assert sample.unrealised == 18  # 20 price P/L minus 2 accrued funding.
    assert sample.equity == 117
    # Close with another 1 fee and settle 2 funding. No double-counting.
    exchange.margin_balance = 116
    position.qty = 0
    position.margin_committed = 0
    deal.status = DealStatus.CLOSED
    await _sample_equity(manager, history)
    sample = history.points[-1]
    assert sample.realised == 16
    assert sample.unrealised == 0
    assert sample.equity == history.start_balance + sample.realised + sample.unrealised
    html = history.render()
    assert 'Realised and unrealised P/L' in html
    assert 'Current drawdown' in html
    assert 'Maximum drawdown' in html
