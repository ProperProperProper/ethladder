from symbot_python.api import paper as paper_module
from symbot_python.api.paper import _session_summary, _svg_equity_curve


def _closed_deal(profit_quote, is_real_win, reason="take_profit", date=1000.0):
    return {
        "status": "closed",
        "unrealized_quote": None,
        "sell_data": {
            "date": date,
            "profit_quote": profit_quote,
            "is_real_win": is_real_win,
            "reason": reason,
        },
    }


def _open_deal(unrealized_quote, margin_used=0.0):
    return {
        "status": "monitoring", "unrealized_quote": unrealized_quote,
        "margin_used": margin_used, "sell_data": None,
    }


def _bots_view(*deals, bot_name="paper-auto-ETHUSDT"):
    return [{"bot_name": bot_name, "deals": list(deals)}]


def test_svg_equity_curve_returns_empty_for_fewer_than_two_points():
    assert _svg_equity_curve([]) == ""
    assert _svg_equity_curve([100.0]) == ""


def test_svg_equity_curve_contains_a_polyline_for_two_or_more_points():
    svg = _svg_equity_curve([100.0, 110.0, 105.0])
    assert svg.startswith("<svg")
    assert "<polyline points=" in svg


def test_session_summary_with_no_closed_deals_returns_zeros_and_no_curve(monkeypatch):
    monkeypatch.setattr(paper_module, "_session_start_balance", 10_000.0)
    monkeypatch.setattr(paper_module, "_session_start_time", 1_700_000_000.0)

    summary = _session_summary(_bots_view(_open_deal(5.0)), free_cash_balance=10_005.0)

    assert summary["total_trades"] == 0
    assert summary["realized_pnl_quote"] == 0
    assert summary["win_rate_percent"] is None
    assert summary["equity_svg"] == ""
    assert summary["unrealized_pnl_quote"] == 5.0
    assert summary["open_deal_count"] == 1


def test_session_summary_computes_realized_pnl_and_win_rate(monkeypatch):
    monkeypatch.setattr(paper_module, "_session_start_balance", 10_000.0)
    monkeypatch.setattr(paper_module, "_session_start_time", 1_700_000_000.0)

    deals = [
        _closed_deal(50.0, True, reason="take_profit", date=1.0),
        _closed_deal(-20.0, False, reason="stop_loss", date=2.0),
        _closed_deal(0.0, False, reason="engine_error", date=3.0),
    ]
    summary = _session_summary(_bots_view(*deals), free_cash_balance=10_030.0)

    assert summary["total_trades"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 2
    assert round(summary["win_rate_percent"], 2) == round(100 / 3, 2)
    assert summary["realized_pnl_quote"] == 30.0
    assert summary["best_trade_quote"] == 50.0
    assert summary["worst_trade_quote"] == -20.0
    assert summary["reason_counts"] == {"take_profit": 1, "stop_loss": 1, "engine_error": 1}
    assert summary["equity_svg"] != ""


def test_session_summary_profit_factor_and_drawdown(monkeypatch):
    monkeypatch.setattr(paper_module, "_session_start_balance", 1000.0)
    monkeypatch.setattr(paper_module, "_session_start_time", 1_700_000_000.0)

    # Balance path: 1000 -> 1100 (peak) -> 1050 -> 1150. Max drawdown is
    # the (peak - trough) dip from 1100 down to 1050 = 50/1100 ~= 4.545%.
    deals = [
        _closed_deal(100.0, True, date=1.0),
        _closed_deal(-50.0, False, date=2.0),
        _closed_deal(100.0, True, date=3.0),
    ]
    summary = _session_summary(_bots_view(*deals), free_cash_balance=1150.0)

    assert summary["profit_factor"] == 200.0 / 50.0
    assert round(summary["max_drawdown_percent"], 2) == round(50 / 1100 * 100, 2)


def test_session_summary_with_no_session_start_balance_skips_curve_but_still_sums_pnl(monkeypatch):
    monkeypatch.setattr(paper_module, "_session_start_balance", None)
    monkeypatch.setattr(paper_module, "_session_start_time", None)

    summary = _session_summary(_bots_view(_closed_deal(10.0, True)), free_cash_balance=10.0)

    assert summary["realized_pnl_quote"] == 10.0
    assert summary["realized_pnl_percent"] is None
    assert summary["equity_svg"] == ""
    assert summary["started_at"] is None


def test_session_summary_balance_includes_margin_locked_in_open_deals(monkeypatch):
    # Regression test: PaperExchangeClient's USDT balance is free cash
    # only — margin_required is debited from it the instant an order
    # fills (paper_client.py's place_market_order). A deal with safety
    # orders filled can lock most of the account's equity as margin,
    # which must still count toward "Balance," or the dashboard reads as
    # a huge loss the moment any deal has more than its base order
    # filled, even though nothing was actually lost.
    monkeypatch.setattr(paper_module, "_session_start_balance", 1000.0)
    monkeypatch.setattr(paper_module, "_session_start_time", 1_700_000_000.0)

    # free cash down to 100 because 880 of the original 1000 is locked as
    # margin in this open deal, which is currently up 20 unrealized.
    summary = _session_summary(
        _bots_view(_open_deal(unrealized_quote=20.0, margin_used=880.0)),
        free_cash_balance=100.0,
    )

    assert summary["current_balance"] == 100.0 + 880.0 + 20.0
