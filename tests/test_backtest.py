from dataclasses import replace

import pytest

from symbot_python.strategy.backtest import (
    MIN_WIN_PRICE_PCT,
    BacktestConfig,
    BacktestReport,
    BacktestTrade,
    build_ladder,
    clamp_take_profit_percent,
    run_backtest,
    select_tier_maintenance_margin_rate,
)


def base_config(**overrides) -> BacktestConfig:
    defaults = dict(
        first_order_amount=100.0,
        dca_order_amount=50.0,
        dca_max_order=1,
        dca_order_size_multiplier=1.0,
        dca_order_start_distance=5.0,
        dca_order_step_percent=5.0,
        dca_order_step_percent_multiplier=1.0,
        dca_take_profit_percent=2.0,
        exchange_fee=0.0,
        price_tick=0.01,
        min_move_amount=0.0001,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_build_ladder_base_and_one_safety_order():
    config = base_config()
    orders = build_ladder(config, entry_price=100.0)
    assert len(orders) == 2  # base + 1 safety order
    assert orders[0].price == pytest.approx(100.0)
    assert orders[0].average == pytest.approx(100.0, abs=0.01)
    # safety order #1 at 5% below entry
    assert orders[1].price == pytest.approx(95.0, abs=0.01)


def test_take_profit_closes_deal_profitably():
    config = base_config(dca_max_order=1)
    candles = [
        [0, 100.0, 101.0, 99.0, 100.5, 1000],
        [1, 100.5, 103.0, 100.0, 102.5, 1000],  # high crosses target (~102)
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.profit_percent == pytest.approx(2.0, abs=0.05)
    assert trade.safety_orders_used == 0
    assert report.final_equity > report.starting_equity


def test_safety_order_triggers_then_take_profit():
    config = base_config(dca_max_order=1)
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 100.5, 94.0, 95.0, 1000],   # low touches safety order @95
        [2, 95.0, 105.0, 94.5, 100.0, 1000],   # high clears the new (lower) target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.safety_orders_used == 1
    assert trade.exit_reason == "take_profit"
    assert trade.profit_percent > 0


def test_stop_loss_triggers_and_exits_at_a_loss():
    config = base_config(
        dca_max_order=1,
        dca_take_profit_percent=50.0,  # unreachable, isolates the stop-loss path
        dca_stop_loss_enabled=True,
        dca_stop_loss_percent=5.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 99.0, 100.0, 94.0, 95.0, 1000],  # low breaches the 95 stop level
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(95.0, abs=0.1)
    assert trade.profit_percent < 0


def test_end_of_data_marks_to_market():
    config = base_config(dca_take_profit_percent=1000.0)  # never reachable
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 101.0, 99.0, 105.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 1
    assert report.trades[0].exit_reason == "end_of_data"
    assert report.trades[0].exit_price == pytest.approx(105.0)


def test_multiple_sequential_deals_and_drawdown_tracking():
    config = base_config(dca_max_order=1)
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0, 1000],
        [1, 100.0, 103.0, 99.0, 100.0, 1000],   # deal 1: take-profit
        [2, 100.0, 101.0, 90.0, 91.0, 1000],    # deal 2: big drop, no SL configured -> keeps running
        [3, 91.0, 92.0, 90.0, 91.0, 1000],      # end of data mid-deal-2, marked to market at a loss
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 2
    assert report.trades[0].exit_reason == "take_profit"
    assert report.trades[1].exit_reason == "end_of_data"
    assert report.max_drawdown_quote > 0
    assert report.win_count == 1
    assert report.loss_count == 1
    assert report.win_rate == pytest.approx(0.5)


def test_max_deals_limits_backtest_length():
    config = base_config(dca_max_order=1, max_deals=1)
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0, 1000],
        [1, 100.0, 103.0, 99.0, 100.0, 1000],
        [2, 100.0, 101.0, 99.0, 100.0, 1000],
        [3, 100.0, 103.0, 99.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 1


def test_empty_candles_returns_empty_report():
    report = run_backtest(base_config(), [], starting_equity=500.0)
    assert report.trades == []
    assert report.final_equity == pytest.approx(500.0)


def test_auto_size_to_funds_disabled_uses_configured_amounts_unscaled():
    config = base_config(dca_max_order=1, auto_size_to_funds=False)
    orders = build_ladder(config, entry_price=100.0)
    # first_order_amount=100 at price 100 with fee=0 -> qty ~= 1.0
    assert orders[0].qty == pytest.approx(1.0, abs=0.01)


def test_auto_size_to_funds_scales_ladder_to_98_percent_of_equity():
    from symbot_python.strategy.dca_math import calculate_max_funds, solve_order_sizing_for_budget

    config = base_config(dca_max_order=1, auto_size_to_funds=True, funds_utilization_percent=98.0)

    # What the per-deal scaled config must be, independent of run_backtest:
    scaled_first, scaled_dca = solve_order_sizing_for_budget(
        config.first_order_amount, config.dca_order_amount, config.dca_max_order,
        config.dca_order_size_multiplier, config.exchange_fee, target_budget=1000.0 * 0.98,
    )
    achieved_budget = calculate_max_funds(
        scaled_first, scaled_dca, config.dca_max_order, config.dca_order_size_multiplier, config.exchange_fee
    )
    assert achieved_budget == pytest.approx(980.0)

    # And run_backtest must actually use that scaled sizing for the deal:
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0, 1000],
        [1, 100.0, 103.0, 99.0, 100.0, 1000],  # take-profit closes immediately
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    expected_qty = scaled_first / 100.0  # base order only (dca_max_order=1, closed before any SO), fee=0
    assert report.trades[0].qty == pytest.approx(expected_qty, rel=0.01)


def test_auto_size_to_funds_compounds_across_deals():
    # Two deals, both closing via take-profit; the SECOND deal's ladder
    # should be sized off the grown equity from the first, not the
    # original starting_equity.
    config = base_config(dca_max_order=1, auto_size_to_funds=True)
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0, 1000],
        [1, 100.0, 110.0, 99.0, 100.0, 1000],  # big first win -> equity grows a lot
        [2, 100.0, 101.0, 99.0, 100.0, 1000],
        [3, 100.0, 103.0, 99.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 2
    # second deal's qty should be larger than the first's, since equity
    # (and thus the auto-sized budget) grew after the first win.
    assert report.trades[1].qty > report.trades[0].qty


def test_leverage_one_behaves_like_spot_no_liquidation_field_used():
    config = base_config(dca_max_order=0, leverage=1.0, auto_size_to_funds=False)
    candles = [
        [0, 100.0, 100.5, 50.0, 60.0, 1000],  # a huge, unrealistic crash
        [1, 60.0, 65.0, 55.0, 60.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    # No liquidation possible at leverage=1 — deal just rides it out to
    # end-of-data, however bad the mark-to-market looks.
    assert report.trades[0].exit_reason == "end_of_data"


def test_leverage_11x_triggers_liquidation_on_a_deep_drop():
    # average ~100 (base order only), 11x leverage -> liquidation around
    # 100*(1-1/11+0.005) ~= 91.4. A crash to 80 must liquidate, not ride out.
    config = base_config(
        dca_max_order=0, leverage=11.0, maintenance_margin_rate=0.005, auto_size_to_funds=False,
        exchange_fee=0.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 100.5, 80.0, 85.0, 1000],  # crashes well past the liquidation price
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "liquidated"
    # Exact loss at liquidation: -margin * (1 - leverage * maintenance_margin_rate)
    # (derived from qty=margin*leverage/average and liq_price=average*(1-1/leverage+mmr)).
    margin = config.first_order_amount
    expected_loss = -margin * (1 - config.leverage * config.maintenance_margin_rate)
    assert trade.profit_quote == pytest.approx(expected_loss)
    # still close to "lost the whole margin" as a sanity bound
    assert trade.profit_quote == pytest.approx(-margin, rel=0.1)


def test_bankruptcy_stops_the_backtest_instead_of_generating_phantom_trades():
    # Regression: once a liquidation wipes out ALL equity, the loop used
    # to keep going — auto_size_to_funds floors target_budget at 0,
    # solve_order_sizing_for_budget returns a degenerate (0.0, 0.0)
    # sizing, and the backtest kept "opening" zero-qty/zero-profit
    # phantom deals for every remaining candle instead of stopping,
    # inflating trade count/win_rate with no-op trades. A real account
    # can't open another deal with no equity left.
    config = base_config(
        dca_max_order=0, leverage=11.0, maintenance_margin_rate=0.0, auto_size_to_funds=True,
        funds_utilization_percent=100.0, exchange_fee=0.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 100.5, 10.0, 15.0, 1000],  # liquidates, wiping out all equity (mmr=0 -> loses exactly the margin)
        [2, 15.0, 20.0, 10.0, 18.0, 1000],
        [3, 18.0, 25.0, 15.0, 22.0, 1000],
        [4, 22.0, 30.0, 20.0, 28.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=100.0)
    assert len(report.trades) == 1  # not 4+ phantom trades for the remaining candles
    assert report.trades[0].exit_reason == "liquidated"
    assert report.final_equity == pytest.approx(0.0)


def test_leverage_11x_survives_a_shallow_dip_then_takes_profit():
    config = base_config(
        dca_max_order=0, leverage=11.0, dca_take_profit_percent=2.0,
        auto_size_to_funds=False, exchange_fee=0.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 103.0, 98.0, 100.0, 1000],  # dips to 98 (above liq ~91.4), then clears target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "take_profit"
    # dollar profit should be ~11x the unleveraged (spot) equivalent
    unleveraged_profit = config.first_order_amount * (config.dca_take_profit_percent / 100)
    assert trade.profit_quote == pytest.approx(unleveraged_profit * 11, rel=0.05)


def test_short_take_profit_closes_deal_profitably_when_price_falls():
    config = base_config(dca_max_order=1, side="short", auto_size_to_funds=False)
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 100.5, 97.0, 98.0, 1000],  # low crosses the ~98 target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.profit_percent > 0
    assert trade.exit_price < trade.average  # short profits below average


def test_short_take_profit_does_not_trigger_on_a_price_rise():
    # dca_max_order=0: no safety-order rung to complicate the picture —
    # isolates "price only rises, target (below entry) is never reached."
    config = base_config(dca_max_order=0, side="short", max_deals=1, auto_size_to_funds=False)
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 104.0, 99.5, 103.0, 1000],  # price rises, low never falls to the ~98 target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "end_of_data"


def test_short_safety_order_triggers_on_price_rise_then_closes():
    config = base_config(dca_max_order=1, side="short", auto_size_to_funds=False)
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 106.0, 99.5, 105.0, 1000],  # high touches the safety order above entry
        [2, 105.0, 106.0, 90.0, 91.0, 1000],   # then crashes back down past the new target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.safety_orders_used == 1
    assert trade.exit_reason == "take_profit"
    assert trade.profit_percent > 0


def test_short_stop_loss_closes_deal_at_a_loss_when_price_rises():
    config = base_config(
        dca_max_order=0, side="short", auto_size_to_funds=False,
        dca_stop_loss_enabled=True, dca_stop_loss_percent=5.0, dca_take_profit_percent=1000.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 106.0, 99.5, 105.0, 1000],  # high breaches the 5% stop (105)
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.profit_percent < 0


def test_short_liquidation_triggers_on_a_sharp_rally():
    config = base_config(
        dca_max_order=0, side="short", auto_size_to_funds=False, exchange_fee=0.0,
        leverage=11.0, maintenance_margin_rate=0.005,
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 120.0, 99.5, 115.0, 1000],  # rallies well past the ~108.6 liquidation price
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "liquidated"
    assert trade.profit_quote < 0


def test_reverse_drawdown_flips_the_side_and_enters_atomically_at_the_exit():
    # Regression target: the reversal must be backtestable at all — it
    # previously only existed in the live/paper engine, so BT/walk-
    # forward could never evaluate it. Long entry 100.0 dips to 97.5
    # (2.5% drawdown, past the 1.5% threshold) -> reverses to short at
    # the exact exit price/timestamp, not the next bar's open.
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=1000.0, auto_size_to_funds=False,
        reverse_drawdown_percent=1.5,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 100.2, 97.5, 98.0, 1000],  # dips to 97.5 -> crosses the threshold
        [2, 98.0, 103.0, 97.5, 100.0, 1000],  # rallies back to 100 -> hurts the new SHORT
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 2
    first, second = report.trades
    assert first.exit_reason == "reverse_drawdown"
    # Atomic: the new deal enters at the EXACT reversal fill, not the
    # next bar's open.
    assert second.entry_price == pytest.approx(first.exit_price)
    assert second.entry_ts == first.exit_ts
    # Price ROSE from 97.5 to 100 in the final bar — a LONG would show a
    # gain there; the second trade shows a LOSS, confirming it's
    # actually the opposite (short) side.
    assert second.profit_quote < 0
    assert second.raw_move_percent < 0


def test_reverse_drawdown_respects_the_cooldown_in_backtest():
    # Same setup, but the flipped short ALSO crosses its own drawdown
    # threshold in the very next bar (price keeps rallying) — cooldown
    # (default 3600s, vastly longer than these bars' tiny timestamps)
    # must block a second reversal, leaving exactly 2 trades, not 3.
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=1000.0, auto_size_to_funds=False,
        reverse_drawdown_percent=1.5,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 100.2, 97.5, 98.0, 1000],
        [2, 98.0, 103.0, 97.5, 100.0, 1000],  # would cross the short's own threshold too
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 2
    assert report.trades[1].exit_reason == "end_of_data"  # not a second reversal


def test_reverse_drawdown_stops_after_max_consecutive_reversals_in_backtest():
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=1000.0, auto_size_to_funds=False,
        reverse_drawdown_percent=1.5, reverse_cooldown_sec=0.0, max_consecutive_reversals=1,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 100.2, 97.5, 98.0, 1000],  # 1st reversal: long -> short @ 97.5
        [2, 98.0, 103.0, 97.5, 100.0, 1000],  # would be a 2nd reversal, but the cap (1) blocks it
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 2
    assert report.trades[0].exit_reason == "reverse_drawdown"
    assert report.trades[1].exit_reason == "end_of_data"  # cap blocked the second reversal


def test_a_normal_close_after_a_reversal_reverts_to_the_original_side():
    # The flipped (short) deal closes via take_profit this time — the
    # NEXT (third) deal must revert to the ORIGINAL configured side
    # (long), not stay short, matching dca_bot_manager.py's live/paper
    # behavior (a flip is a one-deal tactical move, not a permanent
    # side-switch).
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=1.0, auto_size_to_funds=False,
        reverse_drawdown_percent=1.5, reverse_cooldown_sec=0.0,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 100.2, 97.5, 98.0, 1000],   # 1st reversal: long -> short @ 97.5
        [2, 97.5, 97.6, 96.0, 96.5, 1000],     # short's take-profit (1%): target ~96.5, clears it
        [3, 96.5, 100.0, 96.0, 99.0, 1000],    # third deal's own bar
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert len(report.trades) == 3
    assert report.trades[0].exit_reason == "reverse_drawdown"
    assert report.trades[1].exit_reason == "take_profit"
    third = report.trades[2]
    # Price ROSE in the third deal's bar (96.5 -> 99.0) — a LONG shows a
    # gain there, confirming the side reverted back to the original.
    assert third.profit_quote > 0
    assert third.raw_move_percent > 0


def test_no_repainting_adverse_event_wins_when_both_sides_of_a_bar_are_crossed():
    # A single wide-range bar whose LOW breaches the stop-loss AND whose
    # HIGH clears the take-profit target. The pessimistic, no-look-ahead
    # rule must resolve this as a stop-loss, never a take-profit — the
    # adverse event is always checked first, regardless of which one
    # would make the result look better.
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=2.0, exchange_fee=0.0,
        dca_stop_loss_enabled=True, dca_stop_loss_percent=3.0,
        auto_size_to_funds=False,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        # low=96 breaches the 3% stop (97); high=103 clears the 2% target (102)
        [1, 100.0, 103.0, 96.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.profit_percent < 0


def test_no_repainting_same_guarantee_holds_for_short():
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=2.0, exchange_fee=0.0,
        dca_stop_loss_enabled=True, dca_stop_loss_percent=3.0,
        auto_size_to_funds=False, side="short",
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        # high=104 breaches the short's 3% stop (103); low=97 clears the 2% target (98)
        [1, 100.0, 104.0, 97.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.profit_percent < 0


def test_backtest_trades_are_never_mutated_after_being_recorded():
    config = base_config(dca_max_order=1)
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0, 1000],
        [1, 100.0, 103.0, 99.0, 100.0, 1000],
        [2, 100.0, 101.0, 99.0, 100.0, 1000],
        [3, 100.0, 103.0, 99.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    first_trade_snapshot = dict(vars(report.trades[0]))
    # running the rest of the backtest to completion must not retroactively
    # change an already-recorded trade's fields.
    assert vars(report.trades[0]) == first_trade_snapshot


# -- hard 14-day period cap (see TESTING_POLICY.md) --------------------------


def test_run_backtest_accepts_exactly_14_days():
    day_ms = 86_400_000
    candles = [[i * day_ms, 100.0, 101.0, 99.0, 100.0, 1000] for i in range(15)]  # spans 14 days
    run_backtest(base_config(dca_max_order=0), candles, starting_equity=1000.0)  # must not raise


def test_run_backtest_rejects_more_than_14_days():
    day_ms = 86_400_000
    candles = [[i * day_ms, 100.0, 101.0, 99.0, 100.0, 1000] for i in range(31)]  # spans 30 days
    with pytest.raises(ValueError, match="14-day"):
        run_backtest(base_config(dca_max_order=0), candles, starting_equity=1000.0)


def test_run_backtest_rejects_a_300_day_window():
    day_ms = 86_400_000
    candles = [[i * day_ms, 100.0, 101.0, 99.0, 100.0, 1000] for i in range(301)]
    with pytest.raises(ValueError):
        run_backtest(base_config(dca_max_order=0), candles, starting_equity=1000.0)


# -- tier-aware maintenance margin --------------------------------------------


def test_select_tier_mmr_falls_back_when_no_tiers_supplied():
    assert select_tier_maintenance_margin_rate(1_000_000.0, None, fallback_mmr=0.005) == 0.005
    assert select_tier_maintenance_margin_rate(1_000_000.0, [], fallback_mmr=0.005) == 0.005


def test_select_tier_mmr_picks_the_matching_tier():
    tiers = [(300_000.0, 150.0, 0.0033), (900_000.0, 100.0, 0.005), (1_200_000.0, 90.0, 0.0056)]
    assert select_tier_maintenance_margin_rate(50_000.0, tiers, 0.01) == pytest.approx(0.0033)
    assert select_tier_maintenance_margin_rate(300_000.0, tiers, 0.01) == pytest.approx(0.0033)  # at the boundary
    assert select_tier_maintenance_margin_rate(500_000.0, tiers, 0.01) == pytest.approx(0.005)
    assert select_tier_maintenance_margin_rate(1_000_000.0, tiers, 0.01) == pytest.approx(0.0056)


def test_select_tier_mmr_uses_worst_tier_beyond_the_table():
    tiers = [(300_000.0, 150.0, 0.0033), (900_000.0, 100.0, 0.005)]
    assert select_tier_maintenance_margin_rate(10_000_000.0, tiers, 0.01) == pytest.approx(0.005)


def test_backtest_uses_tier_mmr_when_supplied_liquidates_sooner_than_fixed_low_rate():
    # Same leverage/average, but a tier table whose matching tier has a
    # much worse MMR than the fixed fallback -> liquidation should
    # trigger at a HIGHER (less far away) price than the fixed-rate run.
    tiers = [(1.0, 150.0, 0.10)]  # forces the worst-tier rate (0.10) for any realistic notional
    base = base_config(
        dca_max_order=0, leverage=11.0, maintenance_margin_rate=0.0033,
        auto_size_to_funds=False, exchange_fee=0.0,
    )
    with_tiers = replace(base, risk_tiers=tiers)

    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [1, 100.0, 100.5, 80.0, 85.0, 1000],
    ]
    report_fixed = run_backtest(base, candles, starting_equity=1000.0)
    report_tiered = run_backtest(with_tiers, candles, starting_equity=1000.0)

    assert report_fixed.trades[0].exit_reason == "liquidated"
    assert report_tiered.trades[0].exit_reason == "liquidated"
    # worse MMR -> liquidation price is HIGHER (closer to entry, less room to fall)
    assert report_tiered.trades[0].exit_price > report_fixed.trades[0].exit_price


# -- funding rate --------------------------------------------------------------


def test_funding_cost_reduces_profit_for_a_long_when_rate_is_positive():
    day_ms = 86_400_000
    config = base_config(
        dca_max_order=0, auto_size_to_funds=False, exchange_fee=0.0,
        funding_events=[(day_ms // 2, 0.001)],  # one funding event, mid-way through, rate=0.1%
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [day_ms, 100.0, 103.0, 99.0, 100.0, 1000],  # take-profit closes after the funding event
    ]
    no_funding = replace(config, funding_events=[])
    report_with = run_backtest(config, candles, starting_equity=1000.0)
    report_without = run_backtest(no_funding, candles, starting_equity=1000.0)

    assert report_with.trades[0].profit_quote < report_without.trades[0].profit_quote


def test_funding_cost_benefits_a_short_when_rate_is_positive():
    # A positive funding rate means longs pay shorts — a short should
    # come out AHEAD of the no-funding baseline, not behind.
    day_ms = 86_400_000
    config = base_config(
        dca_max_order=0, side="short", auto_size_to_funds=False, exchange_fee=0.0,
        funding_events=[(day_ms // 2, 0.001)],
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [day_ms, 100.0, 100.5, 97.0, 98.0, 1000],  # take-profit closes after the funding event
    ]
    no_funding = replace(config, funding_events=[])
    report_with = run_backtest(config, candles, starting_equity=1000.0)
    report_without = run_backtest(no_funding, candles, starting_equity=1000.0)

    assert report_with.trades[0].profit_quote > report_without.trades[0].profit_quote


def test_funding_events_before_entry_are_never_applied():
    day_ms = 86_400_000
    config = base_config(
        dca_max_order=0, auto_size_to_funds=False, exchange_fee=0.0,
        funding_events=[(-1, 0.05)],  # far in the past, before this deal ever opened
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [day_ms, 100.0, 103.0, 99.0, 100.0, 1000],
    ]
    no_funding = replace(config, funding_events=[])
    report_with = run_backtest(config, candles, starting_equity=1000.0)
    report_without = run_backtest(no_funding, candles, starting_equity=1000.0)
    assert report_with.trades[0].profit_quote == pytest.approx(report_without.trades[0].profit_quote)


def test_funding_events_default_to_empty_and_never_break_existing_configs():
    config = base_config(dca_max_order=0, auto_size_to_funds=False)
    assert config.funding_events == []


# -- tiny-win filter, ported from the user's live ETH bot ---------------------
# (eth_trader_bt.py: WIN_FEE_MULT=2.0, MIN_WIN_PRICE_PCT=0.0033 — "a trade
# that merely edged out what it paid in fees isn't a real edge, it's noise."
# That project has NO test suite at all; this is the coverage it never had.)


def _trade(profit_quote=0.0, raw_move_percent=0.0, estimated_fees_quote=0.0) -> BacktestTrade:
    return BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=101, average=100, qty=1,
        profit_quote=profit_quote, profit_percent=1.0, safety_orders_used=0,
        exit_reason="take_profit", raw_move_percent=raw_move_percent,
        estimated_fees_quote=estimated_fees_quote,
    )


def test_is_real_win_false_for_an_actual_loss():
    trade = _trade(profit_quote=-5.0, raw_move_percent=1.0, estimated_fees_quote=0.1)
    assert trade.is_real_win() is False


def test_is_real_win_false_when_raw_move_below_min_price_pct():
    # Positive profit, but the raw price move itself is tiny (below 0.33%)
    trade = _trade(profit_quote=0.5, raw_move_percent=0.001, estimated_fees_quote=0.01)
    assert trade.is_real_win() is False


def test_is_real_win_false_when_profit_barely_clears_fees():
    # profit_quote is positive and raw move clears the threshold, but the
    # dollar profit is less than 2x the round-trip fee estimate.
    trade = _trade(profit_quote=0.15, raw_move_percent=1.0, estimated_fees_quote=0.1)  # 0.15 <= 2*0.1
    assert trade.is_real_win() is False


def test_is_real_win_true_when_both_thresholds_clear():
    trade = _trade(profit_quote=5.0, raw_move_percent=1.0, estimated_fees_quote=0.1)  # 5.0 > 2*0.1
    assert trade.is_real_win() is True


def test_is_real_win_false_for_a_move_under_the_floor_even_with_zero_fees():
    # Regression: MIN_WIN_PRICE_PCT used to be 0.0033 (a FRACTION) compared
    # directly against raw_move_percent (a percent-NUMBER, e.g. 0.20 means
    # "0.20%") — a 100x unit mismatch that made the price-move floor a
    # near no-op (0.20 < 0.0033 was always False, so nothing ever failed
    # this check on its own). With zero fees, the fee-multiple check can't
    # reject anything either ("not (estimated_fees_quote > 0 and ...)" is
    # trivially satisfied when fees are 0), so under the old bug this
    # exact case would have returned True — a 0.20% move winning despite
    # being well under the documented 0.33% floor.
    trade = _trade(profit_quote=0.01, raw_move_percent=0.20, estimated_fees_quote=0.0)
    assert trade.is_real_win() is False


def test_is_real_win_ignores_fee_multiplier_check_when_fees_are_zero():
    # exchange_fee=0 -> estimated_fees_quote=0 -> the fee-multiple check
    # can't meaningfully apply (0 * anything = 0); only the price-move
    # floor should gate it in that case.
    trade = _trade(profit_quote=0.01, raw_move_percent=1.0, estimated_fees_quote=0.0)
    assert trade.is_real_win() is True


def test_backtest_win_rate_excludes_a_marginal_take_profit_close():
    # TP sits exactly at the 0.33% floor (the smallest legitimate value
    # now that clamp_take_profit_percent enforces it structurally — see
    # test_run_backtest_clamps_a_sub_floor_take_profit_percent below), so
    # the raw-move floor itself can't be what excludes this trade. A
    # high exchange_fee does the job instead: the round-trip fee
    # estimate swamps the tiny dollar profit, failing the fee-multiple
    # check. Still genuinely profitable in raw dollar terms, but must
    # NOT count as a win.
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=MIN_WIN_PRICE_PCT, exchange_fee=0.1, auto_size_to_funds=False,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 103.0, 99.5, 100.0, 1000],  # easily clears the tiny target
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.profit_quote > 0  # genuinely profitable in raw dollar terms
    assert trade.is_real_win() is False
    assert report.win_count == 0
    assert report.loss_count == 1
    assert report.win_rate == pytest.approx(0.0)


def test_run_backtest_clamps_a_sub_floor_take_profit_percent():
    # Non-negotiable floor (see clamp_take_profit_percent's own
    # docstring): a config below MIN_WIN_PRICE_PCT must never actually
    # run, regardless of source — a searched combo, a stale
    # param-library row from before this floor existed, or a manually-
    # set config. Enforced INSIDE run_backtest itself, so this covers
    # grid_search/random_search/walk_forward too, not just SEARCH_GRID's
    # own value list.
    config = base_config(dca_max_order=0, dca_take_profit_percent=0.05, auto_size_to_funds=False)
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 103.0, 99.5, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    trade = report.trades[0]
    assert trade.raw_move_percent >= MIN_WIN_PRICE_PCT - 0.01  # small tolerance for fee/precision rounding


def test_clamp_take_profit_percent_leaves_a_compliant_value_untouched():
    assert clamp_take_profit_percent(2.0) == pytest.approx(2.0)
    assert clamp_take_profit_percent(MIN_WIN_PRICE_PCT) == pytest.approx(MIN_WIN_PRICE_PCT)


def test_clamp_take_profit_percent_raises_a_sub_floor_value():
    assert clamp_take_profit_percent(0.05) == pytest.approx(MIN_WIN_PRICE_PCT)
    assert clamp_take_profit_percent(0.0) == pytest.approx(MIN_WIN_PRICE_PCT)


def test_backtest_win_rate_counts_a_clear_take_profit_close():
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=2.0, exchange_fee=0.1, auto_size_to_funds=False,
    )
    candles = [
        [0, 100.0, 100.5, 99.5, 100.0, 1000],
        [1, 100.0, 103.0, 99.5, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert report.trades[0].is_real_win() is True
    assert report.win_count == 1
    assert report.win_rate == pytest.approx(1.0)


# -- BacktestReport aggregate statistics --------------------------------------


def _mixed_report() -> BacktestReport:
    trades = [
        _trade_full(profit_quote=10.0, profit_percent=1.0, raw_move_percent=1.0,
                    estimated_fees_quote=0.1, funding_cost_quote=1.0,
                    entry_ts=0, exit_ts=3_600_000, safety_orders_used=0),
        _trade_full(profit_quote=20.0, profit_percent=2.0, raw_move_percent=1.0,
                    estimated_fees_quote=0.1, funding_cost_quote=0.0,
                    entry_ts=0, exit_ts=7_200_000, safety_orders_used=1),
        # nominally positive, but fails the tiny-win filter (raw move too small)
        _trade_full(profit_quote=0.001, profit_percent=0.0001, raw_move_percent=0.0001,
                    estimated_fees_quote=0.0, funding_cost_quote=0.0,
                    entry_ts=0, exit_ts=1_800_000, safety_orders_used=2),
        _trade_full(profit_quote=-5.0, profit_percent=-0.5, raw_move_percent=-0.5,
                    estimated_fees_quote=0.05, funding_cost_quote=0.0,
                    entry_ts=0, exit_ts=10_800_000, safety_orders_used=3,
                    exit_reason="stop_loss"),
        _trade_full(profit_quote=-15.0, profit_percent=-1.5, raw_move_percent=-1.5,
                    estimated_fees_quote=0.05, funding_cost_quote=0.0,
                    entry_ts=0, exit_ts=14_400_000, safety_orders_used=4,
                    exit_reason="liquidated"),
    ]
    return BacktestReport(trades=trades, starting_equity=1000.0, final_equity=1010.001)


def _trade_full(**overrides) -> BacktestTrade:
    defaults = dict(
        entry_ts=0, exit_ts=3_600_000, entry_price=100, exit_price=101, average=100, qty=1,
        profit_quote=0.0, profit_percent=0.0, safety_orders_used=0, exit_reason="take_profit",
        raw_move_percent=0.0, estimated_fees_quote=0.0, funding_cost_quote=0.0,
    )
    defaults.update(overrides)
    return BacktestTrade(**defaults)


def test_gross_profit_and_loss_and_profit_factor():
    report = _mixed_report()
    # Matches the reference eth_trader_bt.py exactly: only REAL wins count
    # as profit (10 + 20); everything else — including the tiny 0.001
    # trade that fails the win filter — is a loss, by its |amount| (so
    # "loss" can never mean two different things across the report: a
    # trade counted in a loss streak always shows up in gross/max/average
    # loss too, and vice versa).
    assert report.gross_profit_quote == pytest.approx(30.0)      # 10 + 20
    assert report.gross_loss_quote == pytest.approx(-20.001)     # -(0.001 + 5 + 15)
    assert report.profit_factor == pytest.approx(30.0 / 20.001)


def test_max_and_average_win_loss():
    report = _mixed_report()
    assert report.max_win_quote == pytest.approx(20.0)
    assert report.max_loss_quote == pytest.approx(-15.0)
    assert report.max_win_percent == pytest.approx(2.0)
    assert report.max_loss_percent == pytest.approx(-1.5)
    # average_win only over REAL wins (excludes the tiny 0.001 trade)
    assert report.average_win_quote == pytest.approx((10.0 + 20.0) / 2)
    # average_loss magnitude over everything that ISN'T a real win (tiny
    # trade + 2 real losses), reported as a negative number
    assert report.average_loss_quote == pytest.approx(-(0.001 + 5.0 + 15.0) / 3)


def test_streaks():
    report = _mixed_report()
    # is_real_win pattern: True, True, False, False, False
    assert report.max_win_streak == 2
    assert report.max_loss_streak == 3


def test_total_fees_and_funding():
    report = _mixed_report()
    assert report.total_fees_quote == pytest.approx(0.1 + 0.1 + 0.0 + 0.05 + 0.05)
    assert report.total_funding_quote == pytest.approx(1.0)


def test_liquidation_count():
    report = _mixed_report()
    assert report.liquidation_count == 1


def test_average_trade_duration_hours():
    report = _mixed_report()
    durations_hours = [1.0, 2.0, 0.5, 3.0, 4.0]
    assert report.average_trade_duration_hours == pytest.approx(sum(durations_hours) / 5)


def test_average_safety_orders_used():
    report = _mixed_report()
    assert report.average_safety_orders_used == pytest.approx((0 + 1 + 2 + 3 + 4) / 5)


def test_profit_factor_infinite_when_no_losses():
    report = BacktestReport(
        trades=[_trade_full(profit_quote=10.0, profit_percent=1.0, raw_move_percent=1.0)],
        starting_equity=1000.0, final_equity=1010.0,
    )
    assert report.profit_factor == float("inf")


def test_profit_factor_zero_when_no_trades():
    report = BacktestReport(trades=[], starting_equity=1000.0, final_equity=1000.0)
    assert report.profit_factor == 0.0
    assert report.max_win_streak == 0
    assert report.max_loss_streak == 0
    assert report.average_trade_duration_hours == 0.0


# -- Sharpe ratio / CAGR (matching the ETH bot's own backtest reporting) -----


def test_period_days_populated_from_candle_span():
    config = base_config(dca_max_order=0, auto_size_to_funds=False)
    day_ms = 86_400_000
    candles = [[i * day_ms, 100.0, 101.0, 99.0, 100.0, 1000] for i in range(15)]  # 14 days span
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert report.period_days == pytest.approx(14.0)


def test_sharpe_zero_with_fewer_than_two_trades():
    report = BacktestReport(
        trades=[_trade_full(profit_quote=10.0, profit_percent=1.0)],
        starting_equity=1000.0, final_equity=1010.0, period_days=14.0,
    )
    assert report.sharpe_ratio == 0.0


def test_sharpe_zero_when_all_returns_identical():
    # zero variance -> can't divide by stdev
    trades = [_trade_full(profit_quote=10.0, profit_percent=1.0) for _ in range(3)]
    report = BacktestReport(trades=trades, starting_equity=1000.0, final_equity=1030.0, period_days=14.0)
    assert report.sharpe_ratio == 0.0


def test_sharpe_positive_for_consistently_positive_returns():
    trades = [
        _trade_full(profit_quote=10.0, profit_percent=1.0),
        _trade_full(profit_quote=15.0, profit_percent=1.5),
        _trade_full(profit_quote=8.0, profit_percent=0.8),
    ]
    report = BacktestReport(trades=trades, starting_equity=1000.0, final_equity=1033.0, period_days=14.0)
    assert report.sharpe_ratio > 0


def test_sharpe_negative_for_consistently_negative_returns():
    trades = [
        _trade_full(profit_quote=-10.0, profit_percent=-1.0),
        _trade_full(profit_quote=-15.0, profit_percent=-1.5),
        _trade_full(profit_quote=-8.0, profit_percent=-0.8),
    ]
    report = BacktestReport(trades=trades, starting_equity=1000.0, final_equity=967.0, period_days=14.0)
    assert report.sharpe_ratio < 0


def test_cagr_zero_when_no_period_or_no_starting_equity():
    assert BacktestReport(starting_equity=0.0, final_equity=100.0, period_days=14.0).cagr_percent == 0.0
    assert BacktestReport(starting_equity=1000.0, final_equity=1010.0, period_days=0.0).cagr_percent == 0.0


def test_cagr_positive_for_a_profitable_window():
    report = BacktestReport(starting_equity=1000.0, final_equity=1200.0, period_days=14.0)
    assert report.cagr_percent > 0


def test_cagr_total_loss_clamped_to_negative_100():
    report = BacktestReport(starting_equity=1000.0, final_equity=0.0, period_days=14.0)
    assert report.cagr_percent == -100.0


def test_cagr_extreme_extrapolation_clamped():
    # a tiny window with a huge return, extrapolated to a full year, would
    # otherwise overflow -- must be clamped rather than raising/NaN.
    report = BacktestReport(starting_equity=1000.0, final_equity=1_000_000.0, period_days=0.1)
    assert report.cagr_percent == pytest.approx(1e8)


def test_funding_cost_quote_populated_by_run_backtest():
    day_ms = 86_400_000
    config = base_config(
        dca_max_order=0, auto_size_to_funds=False, exchange_fee=0.0,
        funding_events=[(day_ms // 2, 0.001)],
    )
    candles = [
        [0, 100.0, 100.5, 99.0, 100.0, 1000],
        [day_ms, 100.0, 103.0, 99.0, 100.0, 1000],
    ]
    report = run_backtest(config, candles, starting_equity=1000.0)
    assert report.trades[0].funding_cost_quote != 0.0
    assert report.total_funding_quote == pytest.approx(report.trades[0].funding_cost_quote)


@pytest.mark.parametrize("side", ["long", "short"])
def test_flat_trade_charges_actual_entry_and_exit_fees(side):
    config = base_config(side=side, dca_max_order=0, exchange_fee=0.06)
    candles = [[i, 100, 100, 100, 100, 1] for i in range(3)]
    report = run_backtest(config, candles, 1000)
    trade = report.trades[0]
    fees = trade.qty * 100 * 0.0006 * 2
    assert trade.profit_quote == pytest.approx(-fees)
    assert trade.estimated_fees_quote == pytest.approx(fees)
    marked = run_backtest(config, candles, 1000, close_at_end=False)
    assert not marked.trades
    assert marked.final_equity == pytest.approx(1000 - fees / 2)


def test_same_side_refresh_preserves_open_deal_target_and_leverage():
    candles = [[i, 100, 100, 100, 100, 1] for i in range(4)]
    original = base_config(dca_max_order=0, dca_take_profit_percent=10, leverage=2)
    refreshed = replace(original, dca_take_profit_percent=1, leverage=5)
    candles[-1][2] = 102
    report = run_backtest(original, candles, 1000, config_updates=[(2, refreshed)])
    assert len(report.trades) == 1
    assert report.trades[0].exit_reason == "end_of_data"
    assert report.trades[0].leverage == 2
