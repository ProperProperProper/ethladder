"""Parameter search over BacktestConfig ranges.

Two modes:
- grid_search: try every combination of the given parameter ranges
  against one historical window, rank by a scoring function.
- walk_forward: roll through history in (in-sample optimize) / (out-of-
  sample validate) windows, re-optimizing every step. The chained
  out-of-sample results are the honest estimate of live performance,
  since no window's optimization ever sees the data used to score it —
  this is the standard defense against curve-fitting a strategy to one
  historical period, matching the pattern the user's own
  bybit-kst-walkforward-btc bot already uses for a different strategy.

Pure — built entirely on strategy.backtest.run_backtest, no I/O.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, replace
from typing import Callable, Optional

# Real memory-leak source, confirmed directly (macOS's `leaks`/`heap`
# tools + a gc.get_objects() snapshot): grid_search()/random_search()
# used to accumulate ALL tested candidates' full OptimizationResult —
# including the ENTIRE BacktestReport, every individual BacktestTrade
# included, not just score/config — into one unbounded `results` list.
# A single random_search() call can test up to 100,000 combos (see
# SEARCH_SAMPLES in run_everything.py), and walk_forward() calls it 3x
# per interval across 4 intervals per optimizer cycle — 11,466
# BacktestTrade objects were observed alive in one process snapshot
# taken only ~3 minutes into a fresh run, consistent with this. Fixed
# by keeping only the top MAX_RETAINED_RESULTS candidates BY SCORE at
# any time (a bounded min-heap — heapq.heapreplace when a new result
# beats the current worst-retained one, otherwise the new (lower-
# scoring) result is discarded immediately, never retained at all).
# Deliberately does NOT truncate/mutate any RETAINED result's trades —
# walk_forward()'s liquidation safety gate
# (`next((r for r in ranked if r.report.liquidation_count == 0), ...)`)
# reads a property computed from .trades, and this must stay correct
# for every result the caller can actually see. 2,000 is generously
# larger than any realistic number of consecutive top-scoring
# candidates that could all fail the liquidation gate in practice.
MAX_RETAINED_RESULTS = 2_000

from symbot_python.strategy.backtest import (
    BacktestConfig,
    BacktestReport,
    BacktestTrade,
    _validate_period,
    run_backtest,
)

ScoringFn = Callable[[BacktestReport], float]


def score_total_return(report: BacktestReport) -> float:
    return report.total_profit_quote


def score_return_over_drawdown(report: BacktestReport) -> float:
    """Return / max drawdown — penalizes configs that win big but with
    large equity swings along the way. No trades scores -inf so an
    inactive parameter combination never wins by default; zero drawdown
    with a positive return scores as the raw return (nothing to divide by).
    """
    if not report.trades:
        return float("-inf")
    if report.max_drawdown_quote <= 0:
        return report.total_profit_quote
    return report.total_profit_quote / report.max_drawdown_quote


def score_return_over_drawdown_floored(report: BacktestReport, floor_percent: float = 1.0) -> float:
    """Same idea as score_return_over_drawdown, but the drawdown
    denominator is floored at floor_percent% of starting equity.

    Without a floor, a config that happens to see almost no real
    drawdown in one particular window (rather than being genuinely
    low-risk) produces an artificially enormous score purely from
    dividing by a near-zero number — a real, observed failure mode
    (a single-window search returned a "winner" with 0.00% drawdown,
    profit factor > 200,000, and a 93,000,000%+ extrapolated CAGR — all
    internally consistent given the formulas, all meaningless as a real
    expectation). The floor caps how much reward a config can get purely
    from "got lucky and never drew down" versus an actually-large return.
    """
    if not report.trades:
        return float("-inf")
    floor = report.starting_equity * (floor_percent / 100.0)
    drawdown = max(report.max_drawdown_quote, floor)
    return report.total_profit_quote / drawdown if drawdown > 0 else report.total_profit_quote


def score_lowest_loss_highest_pnl(report: BacktestReport) -> float:
    """Rank by realized loss drag first, then net PnL.

    `gross_loss_quote` is stored as a negative number, so this subtracts
    the absolute loss from net profit. A candidate with less booked loss
    has less drag, and among similar-loss candidates the higher net PnL
    wins naturally.
    """
    if not report.trades:
        return float("-inf")
    return report.total_profit_quote - abs(report.gross_loss_quote)


@dataclass
class ParamGrid:
    """Maps a BacktestConfig field name to the list of values to try.
    Only fields listed here vary; everything else comes from the base
    config passed to grid_search/walk_forward.
    """

    values: dict[str, list[float]]

    def combinations(self) -> list[dict[str, float]]:
        keys = list(self.values.keys())
        if not keys:
            return [{}]
        return [dict(zip(keys, combo)) for combo in itertools.product(*(self.values[k] for k in keys))]

    def total_combinations(self) -> int:
        """Size of the full cartesian product, without materializing it —
        random_search reports this for progress/logging even though it
        never builds the full combinations() list itself."""
        total = 1
        for choices in self.values.values():
            total *= len(choices)
        return total if self.values else 1

    def sample(self, n: int, rng: Optional[random.Random] = None) -> list[dict[str, float]]:
        """Draws n independent random combinations — one value per
        parameter, chosen uniformly at random, WITH replacement (so a
        combination can repeat, and n can exceed total_combinations()) —
        standard random-search sampling, not a no-repeats subset. Never
        materializes the full cartesian product, so this stays cheap even
        when total_combinations() is very large.
        """
        rng = rng or random
        keys = list(self.values.keys())
        if not keys:
            return [{} for _ in range(n)]
        return [{k: rng.choice(self.values[k]) for k in keys} for _ in range(n)]


@dataclass
class OptimizationResult:
    config: BacktestConfig
    report: BacktestReport
    score: float


def grid_search(
    base_config: BacktestConfig,
    grid: ParamGrid,
    candles: list[list[float]],
    starting_equity: float,
    scoring_fn: ScoringFn = score_total_return,
    progress_callback: Optional[Callable[[int, int, Optional["OptimizationResult"]], None]] = None,
    progress_every: int = 500,
) -> list[OptimizationResult]:
    """Backtest every parameter combination once, ranked best-first.

    progress_callback, if given, is called as (combos_done, combos_total,
    best_so_far) every `progress_every` combos and once more at the end —
    this is what lets a long-running caller (e.g.
    scripts/continuous_optimizer.py) report live progress somewhere a
    human can actually see it, rather than the search being a silent
    multi-minute black box.
    """
    _validate_period(candles, context="grid_search")
    combos = grid.combinations()
    total = len(combos)
    heap: list[tuple[float, int, OptimizationResult]] = []  # see MAX_RETAINED_RESULTS's comment
    best_so_far: Optional[OptimizationResult] = None
    for i, combo in enumerate(combos, start=1):
        config = replace(base_config, **combo)
        report = run_backtest(config, candles, starting_equity)
        result = OptimizationResult(config=config, report=report, score=scoring_fn(report))
        if best_so_far is None or result.score > best_so_far.score:
            best_so_far = result
        if len(heap) < MAX_RETAINED_RESULTS:
            heapq.heappush(heap, (result.score, i, result))
        elif result.score > heap[0][0]:
            heapq.heapreplace(heap, (result.score, i, result))
        if progress_callback and (i % progress_every == 0 or i == total):
            progress_callback(i, total, best_so_far)
    heap.sort(key=lambda entry: entry[0], reverse=True)
    return [result for _, _, result in heap]


def random_search(
    base_config: BacktestConfig,
    grid: ParamGrid,
    candles: list[list[float]],
    starting_equity: float,
    n_samples: int,
    scoring_fn: ScoringFn = score_total_return,
    progress_callback: Optional[Callable[[int, int, Optional["OptimizationResult"]], None]] = None,
    progress_every: int = 500,
    rng: Optional[random.Random] = None,
) -> list[OptimizationResult]:
    """Like grid_search, but backtests n_samples independently-drawn
    random combinations instead of the full cartesian product — trades
    exhaustive coverage for a bounded, predictable wall-clock cost per
    run, useful once a grid's full combinations() count makes an
    exhaustive search too slow to finish within one cycle. Same
    signature/progress-callback shape as grid_search so callers (e.g.
    scripts/continuous_optimizer.py) can switch between the two with a
    single call-site change.
    """
    _validate_period(candles, context="random_search")
    combos = grid.sample(n_samples, rng=rng)
    total = len(combos)
    heap: list[tuple[float, int, OptimizationResult]] = []  # see MAX_RETAINED_RESULTS's comment
    best_so_far: Optional[OptimizationResult] = None
    for i, combo in enumerate(combos, start=1):
        config = replace(base_config, **combo)
        report = run_backtest(config, candles, starting_equity)
        result = OptimizationResult(config=config, report=report, score=scoring_fn(report))
        if best_so_far is None or result.score > best_so_far.score:
            best_so_far = result
        if len(heap) < MAX_RETAINED_RESULTS:
            heapq.heappush(heap, (result.score, i, result))
        elif result.score > heap[0][0]:
            heapq.heapreplace(heap, (result.score, i, result))
        if progress_callback and (i % progress_every == 0 or i == total):
            progress_callback(i, total, best_so_far)
    heap.sort(key=lambda entry: entry[0], reverse=True)
    return [result for _, _, result in heap]


@dataclass
class RobustCandidate:
    config: BacktestConfig
    reports: list[BacktestReport]  # one per market-condition window, same order as input
    scores: list[float]  # scoring_fn applied to each report
    worst_score: float
    average_score: float


def robust_search(
    base_config: BacktestConfig,
    grid: ParamGrid,
    windows: list[list[list[float]]],
    starting_equity: float,
    scoring_fn: ScoringFn = score_total_return,
) -> list[RobustCandidate]:
    """Test every parameter combination against EVERY window in `windows`
    (each window should be real historical candles from a genuinely
    different market condition — a calm range, a strong trend, a crash,
    etc — not just different slices of the same continuous history).

    Ranks by WORST-CASE score across windows, not average or best-case:
    a config that does great in a bull run but gets liquidated in a
    crash is not "robust," no matter how good its average looks. This is
    the standard min-max approach to robust parameter selection — it
    directly answers "does this hold up in bad conditions," which
    walk_forward (which only guards against overfitting within one
    continuous history) does not.
    """
    for idx, window in enumerate(windows):
        _validate_period(window, context=f"robust_search (window {idx})")

    candidates: list[RobustCandidate] = []
    for combo in grid.combinations():
        config = replace(base_config, **combo)
        reports = [run_backtest(config, window, starting_equity) for window in windows]
        scores = [scoring_fn(r) for r in reports]
        candidates.append(
            RobustCandidate(
                config=config, reports=reports, scores=scores,
                worst_score=min(scores) if scores else float("-inf"),
                average_score=sum(scores) / len(scores) if scores else float("-inf"),
            )
        )
    candidates.sort(key=lambda c: c.worst_score, reverse=True)
    return candidates


@dataclass
class WalkForwardWindow:
    in_sample_start: int
    in_sample_end: int
    out_sample_start: int
    out_sample_end: int
    best_config: BacktestConfig
    in_sample_score: float
    out_sample_report: BacktestReport


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    combined_out_of_sample_trades: list[BacktestTrade]
    combined_final_equity: float
    starting_equity: float
    continuous_report: Optional[BacktestReport] = None

    @property
    def total_out_of_sample_profit(self) -> float:
        return self.combined_final_equity - self.starting_equity

    @property
    def liquidation_count(self) -> int:
        return sum(1 for t in self.combined_out_of_sample_trades if t.exit_reason == "liquidated")

    @property
    def combined_period_days(self) -> float:
        # Sum of each window's OWN out-of-sample span (each already
        # computed correctly from real candle timestamps by
        # run_backtest) rather than the full input's span — the combined
        # out-of-sample trades don't necessarily tile the whole input
        # contiguously, and this is only ever used for CAGR/Sharpe
        # annualization context, not for scoring or promotion.
        if self.continuous_report is not None:
            return self.continuous_report.period_days
        return sum(w.out_sample_report.period_days for w in self.windows)

    def combined_report(self) -> BacktestReport:
        """A single BacktestReport reconstructed from every window's
        out-of-sample trades, chained in test order — this is the
        honest, walk-forward-validated performance estimate to score
        and promote on, as opposed to any one window's in-sample score
        (which is exactly what walk-forward exists to avoid trusting).
        Drawdown is recomputed against a continuously-tracked equity
        curve across ALL windows (not each window's own independent
        peak), since that's the real risk profile this sequence of
        configs would have shown if run live start to finish.
        """
        if self.continuous_report is not None:
            return self.continuous_report
        equity = self.starting_equity
        peak = equity
        max_dd_quote = 0.0
        max_dd_percent = 0.0
        for trade in self.combined_out_of_sample_trades:
            equity += trade.profit_quote
            peak = max(peak, equity)
            drawdown = peak - equity
            max_dd_quote = max(max_dd_quote, drawdown)
            if peak > 0:
                max_dd_percent = max(max_dd_percent, drawdown / peak * 100)
        return BacktestReport(
            trades=self.combined_out_of_sample_trades,
            starting_equity=self.starting_equity,
            final_equity=self.combined_final_equity,
            max_drawdown_quote=max_dd_quote,
            max_drawdown_percent=max_dd_percent,
            period_days=self.combined_period_days,
        )


class _ContinuousReplay:
    """Replay only the observed OOS prefix with its already-selected winners.

    Replaying a bounded prefix keeps the candle engine deterministic without
    serializing half-completed trades. The old terminal mark is replaced, not
    chained as a synthetic exit. No future winner or price enters training.
    """
    def __init__(self, starting_equity):
        self.starting_equity = starting_equity
        self.candles = []
        self.updates = []
        self.report = BacktestReport(starting_equity=starting_equity, final_equity=starting_equity)

    def advance(self, config, candles):
        previous = self.report
        self.updates.append((len(self.candles), config))
        self.candles.extend(candles)
        self.report = run_backtest(
            self.updates[0][1], self.candles, self.starting_equity,
            config_updates=self.updates, close_at_end=False,
        )
        trades = self.report.trades[len(previous.trades):]
        return BacktestReport(
            trades=trades, starting_equity=previous.final_equity,
            final_equity=self.report.final_equity,
            period_days=(candles[-1][0] - candles[0][0]) / 86_400_000 if len(candles) > 1 else 0,
        )

    def finish(self, windows):
        if self.candles:
            terminal = run_backtest(
                self.updates[0][1], self.candles, self.starting_equity,
                config_updates=self.updates,
            )
            # Charge a final reporting close only once, at the end of all OOS.
            windows[-1].out_sample_report.trades.extend(terminal.trades[len(self.report.trades):])
            windows[-1].out_sample_report.final_equity = terminal.final_equity
            self.report = terminal
        return WalkForwardResult(
            windows=windows, combined_out_of_sample_trades=self.report.trades,
            combined_final_equity=self.report.final_equity,
            starting_equity=self.starting_equity, continuous_report=self.report,
        )


def walk_forward(
    base_config: BacktestConfig,
    grid: ParamGrid,
    candles: list[list[float]],
    starting_equity: float,
    in_sample_bars: int,
    out_sample_bars: int,
    scoring_fn: ScoringFn = score_total_return,
    n_samples: Optional[int] = None,
    rng: Optional[random.Random] = None,
    window_callback: Optional[Callable[[int, "WalkForwardWindow"], None]] = None,
    combo_progress_callback: Optional[Callable[[int, int, Optional["OptimizationResult"]], None]] = None,
    combo_progress_every: int = 500,
) -> WalkForwardResult:
    """n_samples=None (the default) searches each in-sample window
    exhaustively via grid_search, exactly as before. Pass n_samples to
    use random_search instead — needed once the grid is large enough
    that an exhaustive search per window (on top of rolling through
    several windows) would be too slow; see random_search's own
    docstring for the same trade-off in the single-window case.

    window_callback, if given, is called as (window_index, window) each
    time a window completes — lets a long-running caller report
    per-window progress somewhere visible, the same idea as
    grid_search/random_search's own progress_callback but at the
    coarser walk-forward-window granularity.

    combo_progress_callback, if given, is passed straight through to
    each window's own grid_search/random_search call — finer-grained
    progress (individual combos within the CURRENT window) for a caller
    that wants both levels, e.g. a live status display that shows both
    "window 2 of 4" and combo-by-combo progress within it.
    """
    if in_sample_bars <= 0 or out_sample_bars <= 0:
        raise ValueError("in_sample_bars and out_sample_bars must be positive")
    # Validate the FULL input up front — walk_forward slices it into
    # smaller in-sample/out-of-sample windows, each individually compliant
    # via run_backtest's own check, but that alone wouldn't stop a much
    # longer history from being fed in and quietly sliced up. The 14-day
    # cap applies to the whole dataset used for a walk-forward run, not
    # just each window within it.
    _validate_period(candles, context="walk_forward (full input)")

    windows: list[WalkForwardWindow] = []
    equity = starting_equity
    replay = _ContinuousReplay(starting_equity)

    start = 0
    n = len(candles)
    window_index = 0
    while start + in_sample_bars + out_sample_bars <= n:
        in_sample = candles[start : start + in_sample_bars]
        out_start = start + in_sample_bars
        out_sample = candles[out_start : out_start + out_sample_bars]

        if n_samples is not None:
            ranked = random_search(base_config, grid, in_sample, equity, n_samples=n_samples,
                                    scoring_fn=scoring_fn, rng=rng,
                                    progress_callback=combo_progress_callback,
                                    progress_every=combo_progress_every)
        else:
            ranked = grid_search(base_config, grid, in_sample, equity, scoring_fn,
                                  progress_callback=combo_progress_callback,
                                  progress_every=combo_progress_every)
        if not ranked:
            break
        # Non-negotiable safety gate, applied PER WINDOW rather than
        # only once at the end: never spend an out-of-sample test on a
        # config that already showed a liquidation in-sample, no matter
        # how good its score looks otherwise. If nothing in this
        # window's ranked results is safe, there's nothing trustworthy
        # left to walk forward with — stop here rather than either
        # picking an unsafe candidate or silently skipping ahead.
        safe_best = next((r for r in ranked if r.report.liquidation_count == 0), None)
        if safe_best is None:
            break

        out_report = replay.advance(safe_best.config, out_sample)
        window = WalkForwardWindow(
            in_sample_start=start,
            in_sample_end=out_start,
            out_sample_start=out_start,
            out_sample_end=out_start + out_sample_bars,
            best_config=safe_best.config,
            in_sample_score=safe_best.score,
            out_sample_report=out_report,
        )
        windows.append(window)
        if window_callback:
            window_callback(window_index, window)
        equity = out_report.final_equity
        start += out_sample_bars
        window_index += 1

    return replay.finish(windows)


def walk_forward_fixed(
    config: BacktestConfig,
    candles: list[list[float]],
    starting_equity: float,
    in_sample_bars: int,
    out_sample_bars: int,
) -> WalkForwardResult:
    """Same rolling window boundaries as walk_forward, but replays ONE
    FIXED config across every out-of-sample slice instead of
    re-optimizing each window — this is how an already-chosen incumbent
    (e.g. continuous_optimizer.py's current stored winner) must be
    re-scored for its result to be fairly comparable to a walk_forward()
    challenger's combined out-of-sample score.

    Scoring the incumbent via a single full-window backtest instead
    (in-sample-shaped, since that config was originally chosen to fit
    data like it) would inflate its score relative to any challenger and
    make it permanently unbeatable regardless of whether a genuinely
    better config exists — comparing that score against a challenger's
    honest walk-forward combined score is apples to oranges. Using the
    exact same window boundaries here as walk_forward (same
    in_sample_bars/out_sample_bars stepped the same way) makes the two
    combined_report() scores directly comparable.
    """
    if in_sample_bars <= 0 or out_sample_bars <= 0:
        raise ValueError("in_sample_bars and out_sample_bars must be positive")
    _validate_period(candles, context="walk_forward_fixed (full input)")

    windows: list[WalkForwardWindow] = []
    equity = starting_equity
    replay = _ContinuousReplay(starting_equity)

    start = 0
    n = len(candles)
    window_index = 0
    while start + in_sample_bars + out_sample_bars <= n:
        out_start = start + in_sample_bars
        out_sample = candles[out_start : out_start + out_sample_bars]

        out_report = replay.advance(config, out_sample)
        window = WalkForwardWindow(
            in_sample_start=start,
            in_sample_end=out_start,
            out_sample_start=out_start,
            out_sample_end=out_start + out_sample_bars,
            best_config=config,
            in_sample_score=0.0,  # not re-optimized per window — nothing to report here
            out_sample_report=out_report,
        )
        windows.append(window)
        equity = out_report.final_equity
        start += out_sample_bars
        window_index += 1

    return replay.finish(windows)
