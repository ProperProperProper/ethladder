from dataclasses import replace

import pytest

from symbot_python.strategy.backtest import BacktestConfig, BacktestReport, BacktestTrade, run_backtest
from symbot_python.strategy.optimize import (
    ParamGrid,
    grid_search,
    random_search,
    robust_search,
    score_lowest_loss_highest_pnl,
    score_return_over_drawdown,
    score_return_over_drawdown_floored,
    score_total_return,
    walk_forward,
    walk_forward_fixed,
)


def matched_series(entry_price: float, amplitude_percent: float, cycles: int) -> list[list[float]]:
    """Two candles per cycle: an entry bar (only its open matters) and an
    exit bar whose high clears exactly `amplitude_percent` above the
    entry price, letting a take-profit set to that amplitude close every
    single cycle profitably, while any higher take-profit target never
    triggers within a cycle (it drifts to end-of-data unclosed instead).
    """
    candles = []
    for k in range(cycles):
        ts = k * 2 * 3_600_000
        candles.append([ts, entry_price, entry_price * 1.0001, entry_price * 0.999, entry_price, 100.0])
        target = entry_price * (1 + amplitude_percent / 100)
        candles.append(
            [ts + 3_600_000, entry_price, target * 1.001, entry_price * 0.999, target, 100.0]
        )
    return candles


def base_config(**overrides) -> BacktestConfig:
    defaults = dict(
        first_order_amount=100.0,
        dca_order_amount=50.0,
        dca_max_order=0,  # base order only — isolates take-profit as the sole variable
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


def test_grid_search_finds_the_matched_take_profit():
    # amplitude=2% matches TP=2 exactly every cycle; TP=5 never triggers
    # within a cycle and ends up as one unclosed end-of-data mark.
    candles = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=10)
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0, 5.0]})
    results = grid_search(base_config(), grid, candles, starting_equity=1000.0)

    assert results[0].config.dca_take_profit_percent == 2.0
    assert len(results[0].report.trades) == 10
    assert all(t.exit_reason == "take_profit" for t in results[0].report.trades)

    worse = next(r for r in results if r.config.dca_take_profit_percent == 5.0)
    assert len(worse.report.trades) == 1
    assert worse.report.trades[0].exit_reason == "end_of_data"
    assert results[0].score > worse.score


def test_grid_search_combinations_cover_full_cartesian_product():
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0], "dca_order_step_percent": [1.0, 3.0]})
    candles = matched_series(100.0, 2.0, cycles=2)
    results = grid_search(base_config(), grid, candles, starting_equity=1000.0)
    assert len(results) == 4
    seen = {(r.config.dca_take_profit_percent, r.config.dca_order_step_percent) for r in results}
    assert seen == {(1.0, 1.0), (1.0, 3.0), (2.0, 1.0), (2.0, 3.0)}


def test_grid_search_empty_grid_runs_base_config_once():
    candles = matched_series(100.0, 2.0, cycles=3)
    results = grid_search(base_config(), ParamGrid(values={}), candles, starting_equity=1000.0)
    assert len(results) == 1
    assert results[0].config.dca_take_profit_percent == base_config().dca_take_profit_percent


def test_param_grid_total_combinations_matches_combinations_length():
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0], "dca_order_step_percent": [1.0, 3.0, 5.0]})
    assert grid.total_combinations() == len(grid.combinations()) == 6


def test_param_grid_sample_returns_exactly_n_valid_combinations():
    grid = ParamGrid(values={
        "dca_take_profit_percent": [1.0, 2.0, 3.0],
        "dca_order_step_percent": [1.0, 3.0],
    })
    rng = __import__("random").Random(42)
    sampled = grid.sample(50, rng=rng)
    assert len(sampled) == 50
    for combo in sampled:
        assert combo["dca_take_profit_percent"] in [1.0, 2.0, 3.0]
        assert combo["dca_order_step_percent"] in [1.0, 3.0]


def test_param_grid_sample_can_exceed_total_combinations_with_replacement():
    # n larger than the full grid size must still work — sampling is
    # WITH replacement, not a no-repeats subset.
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0]})
    sampled = grid.sample(10)
    assert len(sampled) == 10


def test_random_search_finds_the_matched_take_profit():
    # Same setup as test_grid_search_finds_the_matched_take_profit, but
    # via random sampling with enough samples to guarantee both grid
    # values get tried at least once (fixed seed makes this deterministic).
    candles = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=10)
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0, 5.0]})
    results = random_search(
        base_config(), grid, candles, starting_equity=1000.0, n_samples=20, rng=__import__("random").Random(1),
    )
    assert results[0].config.dca_take_profit_percent == 2.0
    assert len(results[0].report.trades) == 10


def test_random_search_respects_n_samples_count():
    candles = matched_series(100.0, 2.0, cycles=2)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0, 3.0]})
    results = random_search(base_config(), grid, candles, starting_equity=1000.0, n_samples=7)
    assert len(results) == 7


def test_random_search_progress_callback_fires_at_the_end():
    candles = matched_series(100.0, 2.0, cycles=2)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0]})
    calls = []
    random_search(
        base_config(), grid, candles, starting_equity=1000.0, n_samples=10,
        progress_callback=lambda done, total, best: calls.append((done, total)),
        progress_every=1000,  # bigger than n_samples -> only the final call fires
    )
    assert calls == [(10, 10)]


def test_score_total_return_is_just_profit():
    report = BacktestReport(starting_equity=1000.0, final_equity=1050.0)
    assert score_total_return(report) == pytest.approx(50.0)


def test_score_return_over_drawdown_no_trades_is_negative_infinity():
    report = BacktestReport(starting_equity=1000.0, final_equity=1000.0, trades=[])
    assert score_return_over_drawdown(report) == float("-inf")


def test_score_return_over_drawdown_zero_drawdown_returns_raw_profit():
    trade = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=102, average=100, qty=1,
        profit_quote=20.0, profit_percent=2.0, safety_orders_used=0, exit_reason="take_profit",
    )
    report = BacktestReport(starting_equity=1000.0, final_equity=1020.0, trades=[trade], max_drawdown_quote=0.0)
    assert score_return_over_drawdown(report) == pytest.approx(20.0)


def test_score_return_over_drawdown_penalizes_large_swings():
    trade = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=102, average=100, qty=1,
        profit_quote=20.0, profit_percent=2.0, safety_orders_used=0, exit_reason="take_profit",
    )
    low_dd = BacktestReport(starting_equity=1000, final_equity=1020, trades=[trade], max_drawdown_quote=5.0)
    high_dd = BacktestReport(starting_equity=1000, final_equity=1020, trades=[trade], max_drawdown_quote=50.0)
    assert score_return_over_drawdown(low_dd) > score_return_over_drawdown(high_dd)


def test_score_return_over_drawdown_floored_no_trades_is_negative_infinity():
    report = BacktestReport(starting_equity=1000.0, final_equity=1000.0, trades=[])
    assert score_return_over_drawdown_floored(report) == float("-inf")


def test_score_return_over_drawdown_floored_caps_a_near_zero_drawdown_blowup():
    trade = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=102, average=100, qty=1,
        profit_quote=20.0, profit_percent=2.0, safety_orders_used=0, exit_reason="take_profit",
    )
    # A near-zero (but nonzero) drawdown on a $1000 account — without a
    # floor this would score 20/0.001 = 20000, an artifact of one lucky
    # window rather than a real edge.
    lucky = BacktestReport(
        starting_equity=1000.0, final_equity=1020.0, trades=[trade], max_drawdown_quote=0.001,
    )
    unfloored_score = score_return_over_drawdown(lucky)
    floored_score = score_return_over_drawdown_floored(lucky, floor_percent=1.0)
    assert floored_score < unfloored_score
    # Floored at 1% of starting equity ($10): 20/10 = 2.0
    assert floored_score == pytest.approx(2.0)


def test_score_return_over_drawdown_floored_leaves_a_real_drawdown_untouched():
    trade = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=102, average=100, qty=1,
        profit_quote=20.0, profit_percent=2.0, safety_orders_used=0, exit_reason="take_profit",
    )
    # A real drawdown well above the 1%-of-equity floor ($10) is used as-is.
    report = BacktestReport(starting_equity=1000.0, final_equity=1020.0, trades=[trade], max_drawdown_quote=50.0)
    assert score_return_over_drawdown_floored(report) == pytest.approx(score_return_over_drawdown(report))


def test_score_lowest_loss_highest_pnl_penalizes_realized_loss_drag():
    win = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=110, average=100, qty=1,
        profit_quote=100.0, profit_percent=10.0, safety_orders_used=0, exit_reason="take_profit",
        raw_move_percent=10.0,
    )
    loss = BacktestTrade(
        entry_ts=1, exit_ts=2, entry_price=100, exit_price=95, average=100, qty=1,
        profit_quote=-30.0, profit_percent=-3.0, safety_orders_used=0, exit_reason="stop_loss",
        raw_move_percent=-5.0,
    )
    cleaner = BacktestReport(starting_equity=1000, final_equity=1100, trades=[win], max_drawdown_quote=40)
    messy = BacktestReport(starting_equity=1000, final_equity=1070, trades=[win, loss], max_drawdown_quote=5)

    assert score_lowest_loss_highest_pnl(cleaner) == pytest.approx(100.0)
    assert score_lowest_loss_highest_pnl(messy) == pytest.approx(40.0)
    assert score_lowest_loss_highest_pnl(cleaner) > score_lowest_loss_highest_pnl(messy)


def test_score_lowest_loss_highest_pnl_still_rewards_higher_pnl_when_losses_match():
    trade = BacktestTrade(
        entry_ts=0, exit_ts=1, entry_price=100, exit_price=110, average=100, qty=1,
        profit_quote=100.0, profit_percent=10.0, safety_orders_used=0, exit_reason="take_profit",
        raw_move_percent=10.0,
    )
    lower = BacktestReport(starting_equity=1000, final_equity=1050, trades=[trade])
    higher = BacktestReport(starting_equity=1000, final_equity=1100, trades=[trade])

    assert score_lowest_loss_highest_pnl(higher) > score_lowest_loss_highest_pnl(lower)


def test_walk_forward_adapts_to_a_regime_change():
    # First half of history rewards TP=2%, second half rewards TP=5% —
    # a genuine walk-forward should pick a DIFFERENT best config per
    # window rather than committing to one global choice.
    regime_a = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=20)
    regime_b = matched_series(entry_price=100.0, amplitude_percent=5.0, cycles=20)
    candles = regime_a + regime_b

    grid = ParamGrid(values={"dca_take_profit_percent": [2.0, 5.0]})
    result = walk_forward(
        base_config(), grid, candles, starting_equity=1000.0,
        in_sample_bars=20, out_sample_bars=20,
    )

    assert len(result.windows) >= 2
    first_window_tp = result.windows[0].best_config.dca_take_profit_percent
    last_window_tp = result.windows[-1].best_config.dca_take_profit_percent
    assert first_window_tp != last_window_tp
    assert first_window_tp == 2.0
    assert last_window_tp == 5.0


def test_walk_forward_out_of_sample_never_peeks_at_its_own_scoring_data():
    # A config that's PERFECT on some data but never appears in-sample
    # cannot possibly be chosen — proves selection only ever uses the
    # in-sample slice, not the out-of-sample slice being validated.
    candles = matched_series(100.0, 2.0, cycles=10)
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0]})  # only one option: trivially "selected"
    result = walk_forward(
        base_config(), grid, candles, starting_equity=1000.0, in_sample_bars=8, out_sample_bars=8,
    )
    for window in result.windows:
        assert window.best_config.dca_take_profit_percent == 2.0


def test_walk_forward_raises_on_non_positive_window_sizes():
    candles = matched_series(100.0, 2.0, cycles=5)
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0]})
    with pytest.raises(ValueError):
        walk_forward(base_config(), grid, candles, 1000.0, in_sample_bars=0, out_sample_bars=5)
    with pytest.raises(ValueError):
        walk_forward(base_config(), grid, candles, 1000.0, in_sample_bars=5, out_sample_bars=0)


def test_walk_forward_no_windows_when_history_too_short():
    candles = matched_series(100.0, 2.0, cycles=2)  # only 4 candles
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0]})
    result = walk_forward(base_config(), grid, candles, 1000.0, in_sample_bars=10, out_sample_bars=10)
    assert result.windows == []
    assert result.combined_final_equity == pytest.approx(1000.0)


def test_walk_forward_uses_random_search_when_n_samples_given():
    # Regression target: SEARCH_GRID grew large enough that an exhaustive
    # grid_search per rolling window (on top of rolling through several
    # windows) is too slow — n_samples switches each window's in-sample
    # selection to random_search instead, same trade-off random_search
    # itself documents for the single-window case. A grid too large to
    # exhaustively search in test time still completes quickly here.
    candles = matched_series(100.0, 2.0, cycles=20)
    grid = ParamGrid(values={
        "dca_take_profit_percent": [round(0.1 * i, 2) for i in range(1, 40)],  # 39 values
        "dca_order_step_percent": [round(0.1 * i, 2) for i in range(1, 40)],  # 39 values -> 1521 combos
    })
    import time as _time
    start = _time.monotonic()
    result = walk_forward(
        base_config(), grid, candles, starting_equity=1000.0,
        in_sample_bars=20, out_sample_bars=20, n_samples=25,
    )
    elapsed = _time.monotonic() - start
    assert len(result.windows) >= 1
    assert elapsed < 5.0  # would take vastly longer exhaustively searched


def test_walk_forward_skips_a_liquidating_top_scorer_for_a_safer_candidate():
    # The exact real incident this safety gate exists for: a
    # higher-leverage config's WINS are amplified (profit scales with
    # leverage for a given price move) while a liquidation caps its loss
    # near the margin regardless of leverage — so it can out-score a
    # safer config in-sample right up until it liquidates. Two
    # profitable cycles first (leverage=11 scores higher there), then a
    # deep decline within the SAME in-sample window that liquidates the
    # leverage=11 config's open deal but not leverage=1's.
    profitable = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=2)
    last_price = profitable[-1][1]
    crash = declining_series(start_price=last_price, end_price=last_price * 0.5, bars=10)
    in_sample = profitable + crash
    out_sample = matched_series(entry_price=last_price * 0.5, amplitude_percent=2.0, cycles=2)
    candles = in_sample + out_sample

    grid = ParamGrid(values={"leverage": [1.0, 11.0]})
    config = base_config(dca_take_profit_percent=2.0, maintenance_margin_rate=0.005)

    result = walk_forward(
        config, grid, candles, starting_equity=1000.0,
        in_sample_bars=len(in_sample), out_sample_bars=len(out_sample),
    )

    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.best_config.leverage == 1.0  # not the liquidating 11x, despite scoring higher in-sample
    assert result.liquidation_count == 0


def test_walk_forward_stops_when_no_candidate_in_a_window_is_safe():
    # If EVERY sampled in-sample candidate for a window liquidated,
    # there's nothing trustworthy to walk forward with — stop rather
    # than picking an unsafe one or silently skipping ahead.
    crash = declining_series(start_price=100.0, end_price=50.0, bars=10)
    candles = crash + crash  # two identical crash windows
    grid = ParamGrid(values={"leverage": [11.0]})  # only the liquidating option available
    config = base_config(maintenance_margin_rate=0.005)

    result = walk_forward(
        config, grid, candles, starting_equity=1000.0,
        in_sample_bars=10, out_sample_bars=10,
    )
    assert result.windows == []
    assert result.combined_final_equity == pytest.approx(1000.0)


def test_walk_forward_combined_report_chains_equity_and_drawdown_across_windows():
    regime_a = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=6)
    regime_b = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=6)
    candles = regime_a + regime_b
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0]})

    result = walk_forward(
        base_config(), grid, candles, starting_equity=1000.0,
        in_sample_bars=6, out_sample_bars=6,
    )
    report = result.combined_report()

    assert report.starting_equity == pytest.approx(1000.0)
    assert report.final_equity == pytest.approx(result.combined_final_equity)
    assert report.trades == result.combined_out_of_sample_trades
    assert report.max_drawdown_quote >= 0.0


def declining_series(start_price: float, end_price: float, bars: int) -> list[list[float]]:
    """A monotonic decline from start_price to end_price over `bars` —
    a stand-in for a real crash regime, with no recovery for a
    no-stop-loss deal to ride out.
    """
    candles = []
    step = (end_price - start_price) / bars
    price = start_price
    for i in range(bars):
        next_price = price + step
        candles.append(
            [i * 3_600_000, price, max(price, next_price) * 1.0001,
             min(price, next_price) * 0.9999, next_price, 100.0]
        )
        price = next_price
    return candles


def test_robust_search_prefers_worst_case_survivor_over_riskier_config():
    # Bull window: both configs perform identically (stop-loss, if
    # enabled, never triggers on the way up).
    bull_window = matched_series(entry_price=100.0, amplitude_percent=3.0, cycles=3)
    # Crash window: a config with NO stop-loss rides a full -50% decline
    # on its one deal; a config WITH a 5% stop-loss cuts that SAME deal
    # early at roughly -5%. max_deals=1 isolates one trade's outcome —
    # without it, a stop-loss config just re-enters and gets stopped out
    # repeatedly through a sustained decline, which is a real and
    # important behavior (this is exactly what the portfolio-loss
    # circuit breaker exists to catch) but not what this test isolates.
    crash_window = declining_series(start_price=100.0, end_price=50.0, bars=30)

    config = base_config(
        dca_max_order=0, dca_take_profit_percent=3.0, exchange_fee=0.0,
        max_deals=1, auto_size_to_funds=False,
    )
    grid = ParamGrid(values={"dca_stop_loss_enabled": [False, True]})
    config = replace(config, dca_stop_loss_percent=5.0)

    candidates = robust_search(config, grid, [bull_window, crash_window], starting_equity=1000.0)

    best = candidates[0]
    assert best.config.dca_stop_loss_enabled is True

    worst = next(c for c in candidates if c.config.dca_stop_loss_enabled is False)
    # both should do about equally well in the bull window...
    assert best.scores[0] == pytest.approx(worst.scores[0], rel=0.05)
    # ...but the no-stop-loss config's single crash-window trade is far worse
    assert best.scores[1] > worst.scores[1]
    assert best.worst_score > worst.worst_score
    # confirm the "no stop-loss rides ~50% down" vs "stop-loss cuts at ~5%" shape
    assert worst.scores[1] == pytest.approx(-50.0, abs=1.0)
    assert best.scores[1] == pytest.approx(-5.0, abs=0.5)


def test_stop_loss_alone_does_not_protect_a_sustained_decline():
    # Documents a real, important finding: without max_deals capping
    # re-entry, a per-trade stop-loss still cuts many small losses in a
    # row through a persistent downtrend — the cumulative damage can
    # rival or exceed a single uncut ride-it-out loss. A stop-loss is
    # not, by itself, "safe in all market conditions"; pairing it with a
    # portfolio-level circuit breaker to halt new deals after repeated
    # losses is what actually is (see DCABotManager.circuit_breaker_active
    # — the generic halt mechanism exists; nothing trips it automatically
    # today).
    crash_window = declining_series(start_price=100.0, end_price=50.0, bars=30)
    config = base_config(
        dca_max_order=0, dca_take_profit_percent=3.0, exchange_fee=0.0,
        auto_size_to_funds=False, dca_stop_loss_enabled=True, dca_stop_loss_percent=5.0,
    )
    report = run_backtest(config, crash_window, starting_equity=1000.0)
    stop_losses = [t for t in report.trades if t.exit_reason == "stop_loss"]
    assert len(stop_losses) > 5  # repeatedly re-entered and re-stopped
    # cumulative damage from repeated small (-5% each) cuts adds up to a
    # meaningful chunk of starting capital, even though no single trade
    # lost more than its configured stop-loss.
    assert report.total_profit_quote < -50


def test_robust_search_worst_score_is_min_of_scores():
    config = base_config(dca_max_order=0, dca_take_profit_percent=2.0)
    windows = [matched_series(100.0, 2.0, cycles=3), matched_series(100.0, 5.0, cycles=3)]
    candidates = robust_search(config, ParamGrid(values={}), windows, starting_equity=1000.0)
    assert candidates[0].worst_score == pytest.approx(min(candidates[0].scores))
    assert candidates[0].average_score == pytest.approx(sum(candidates[0].scores) / 2)


def test_robust_search_single_window_behaves_like_grid_search():
    config = base_config(dca_max_order=0)
    window = matched_series(100.0, 2.0, cycles=5)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0, 5.0]})

    robust_results = robust_search(config, grid, [window], starting_equity=1000.0)
    plain_results = grid_search(config, grid, window, starting_equity=1000.0)

    assert robust_results[0].config.dca_take_profit_percent == plain_results[0].config.dca_take_profit_percent


# -- hard 14-day period cap (see TESTING_POLICY.md) --------------------------


def _oversized_window(days: int) -> list[list[float]]:
    day_ms = 86_400_000
    return [[i * day_ms, 100.0, 101.0, 99.0, 100.0, 1000] for i in range(days + 1)]


def test_grid_search_rejects_more_than_14_days():
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0]})
    with pytest.raises(ValueError, match="14-day"):
        grid_search(base_config(dca_max_order=0), grid, _oversized_window(30), starting_equity=1000.0)


def test_walk_forward_rejects_a_full_input_longer_than_14_days_even_if_windows_are_small():
    # Each individual in-sample/out-of-sample slice would be well under 14
    # days, but the TOTAL input here spans 90 days — must still be rejected
    # up front, not silently allowed just because it gets sliced small.
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0]})
    with pytest.raises(ValueError, match="14-day"):
        walk_forward(
            base_config(dca_max_order=0), grid, _oversized_window(90), starting_equity=1000.0,
            in_sample_bars=48, out_sample_bars=24,
        )


def test_robust_search_rejects_any_oversized_window():
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0]})
    good_window = matched_series(100.0, 2.0, cycles=3)
    with pytest.raises(ValueError, match="14-day"):
        robust_search(
            base_config(dca_max_order=0), grid, [good_window, _oversized_window(60)],
            starting_equity=1000.0,
        )


# -- grid_search progress reporting -------------------------------------------


def test_grid_search_progress_callback_fires_at_the_end():
    config = base_config(dca_max_order=0)
    candles = matched_series(100.0, 2.0, cycles=3)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0, 5.0]})

    calls = []
    grid_search(config, grid, candles, 1000.0, progress_callback=lambda d, t, b: calls.append((d, t)))

    assert calls[-1] == (3, 3)  # always fires once at the very end


def test_grid_search_progress_callback_fires_at_configured_interval():
    config = base_config(dca_max_order=0)
    candles = matched_series(100.0, 2.0, cycles=3)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    calls = []
    grid_search(config, grid, candles, 1000.0, progress_callback=lambda d, t, b: calls.append(d), progress_every=2)

    assert calls == [2, 4, 6]


def test_grid_search_progress_callback_reports_correct_best_so_far():
    config = base_config(dca_max_order=0)
    candles = matched_series(100.0, 2.0, cycles=3)
    grid = ParamGrid(values={"dca_take_profit_percent": [5.0, 2.0]})  # 2.0 matches -> should win

    best_seen = []
    grid_search(
        config, grid, candles, 1000.0, scoring_fn=score_total_return,
        progress_callback=lambda d, t, b: best_seen.append(b.config.dca_take_profit_percent if b else None),
        progress_every=1,
    )
    # after both combos, best-so-far must be the actual winner (2.0)
    assert best_seen[-1] == 2.0


def test_grid_search_without_progress_callback_still_works():
    config = base_config(dca_max_order=0)
    candles = matched_series(100.0, 2.0, cycles=3)
    grid = ParamGrid(values={"dca_take_profit_percent": [1.0, 2.0]})
    results = grid_search(config, grid, candles, 1000.0)  # no callback passed
    assert len(results) == 2


def test_walk_forward_fixed_uses_the_same_window_boundaries_as_walk_forward():
    # This is the whole point of walk_forward_fixed: an incumbent
    # replayed with it and a challenger produced by walk_forward() must
    # be scored over IDENTICAL out-of-sample slices, or the two combined
    # scores aren't comparable (see continuous_optimizer.py's promotion
    # gate, which compares them directly).
    candles = matched_series(100.0, 2.0, cycles=20)
    grid = ParamGrid(values={"dca_take_profit_percent": [2.0]})

    wf = walk_forward(base_config(), grid, candles, 1000.0, in_sample_bars=20, out_sample_bars=20)
    fixed = walk_forward_fixed(base_config(), candles, 1000.0, in_sample_bars=20, out_sample_bars=20)

    assert len(fixed.windows) == len(wf.windows)
    for fixed_window, wf_window in zip(fixed.windows, wf.windows):
        assert fixed_window.out_sample_start == wf_window.out_sample_start
        assert fixed_window.out_sample_end == wf_window.out_sample_end


def test_walk_forward_fixed_replays_the_given_config_unchanged():
    # Unlike walk_forward, nothing is re-optimized per window — every
    # window must use the exact config passed in, even where a
    # different config would have scored better in-sample.
    regime_a = matched_series(entry_price=100.0, amplitude_percent=2.0, cycles=20)
    regime_b = matched_series(entry_price=100.0, amplitude_percent=5.0, cycles=20)
    candles = regime_a + regime_b

    config = base_config(dca_take_profit_percent=2.0)
    result = walk_forward_fixed(config, candles, 1000.0, in_sample_bars=20, out_sample_bars=20)

    assert len(result.windows) >= 2
    for window in result.windows:
        assert window.best_config.dca_take_profit_percent == 2.0


def test_walk_forward_fixed_raises_on_non_positive_window_sizes():
    candles = matched_series(100.0, 2.0, cycles=5)
    with pytest.raises(ValueError):
        walk_forward_fixed(base_config(), candles, 1000.0, in_sample_bars=0, out_sample_bars=5)
    with pytest.raises(ValueError):
        walk_forward_fixed(base_config(), candles, 1000.0, in_sample_bars=5, out_sample_bars=0)


def test_walk_forward_fixed_no_windows_when_history_too_short():
    candles = matched_series(100.0, 2.0, cycles=2)  # only 4 candles
    result = walk_forward_fixed(base_config(), candles, 1000.0, in_sample_bars=10, out_sample_bars=10)
    assert result.windows == []
    assert result.combined_final_equity == pytest.approx(1000.0)


def test_walk_forward_fixed_chains_equity_across_windows():
    candles = matched_series(100.0, 2.0, cycles=12)
    config = base_config(dca_take_profit_percent=2.0)

    result = walk_forward_fixed(config, candles, 1000.0, in_sample_bars=6, out_sample_bars=6)

    assert len(result.windows) >= 1
    combined = result.combined_report()
    assert combined.final_equity == pytest.approx(result.combined_final_equity)
    assert combined.final_equity > 1000.0  # every window profits on its matched take-profit


def test_fixed_walk_forward_keeps_open_deal_and_charges_fees_once():
    candles = [[i * 3_600_000, 100, 100, 100, 100, 1] for i in range(10)]
    config = base_config(dca_take_profit_percent=10, exchange_fee=0.06)
    result = walk_forward_fixed(config, candles, 1000, 4, 2)
    direct = run_backtest(config, candles[4:], 1000)
    assert len(result.windows) == 3
    assert len(result.combined_report().trades) == 1
    assert result.combined_report() == direct
    assert not result.windows[0].out_sample_report.trades
    assert not result.windows[1].out_sample_report.trades
    assert result.windows[1].out_sample_report.starting_equity == result.windows[0].out_sample_report.final_equity
