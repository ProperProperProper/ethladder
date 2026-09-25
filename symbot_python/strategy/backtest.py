"""Deterministic bar-replay backtester for the DCA strategy.

Reuses the exact same pure decision modules the live/paper engine will
use (dca_math, stop_loss) so a backtest result reflects the real trading
logic, not a separate approximation of it. Takes OHLCV candles in the
same shape BybitClient.get_kline/PaperExchangeClient.get_kline return
([timestamp_ms, open, high, low, close, volume], oldest first) — so the
exact same historical data used here can later be fetched live for
paper/live trading.

Simplifications versus the live engine (documented, not hidden):
- One candle = one evaluation step. A stop-loss/take-profit trigger is
  detected using the candle's low/high (the most price could have moved
  within the bar) and filled at the exact trigger level, ignoring
  intra-bar slippage — a conservative-but-optimistic assumption common to
  bar-based backtests.
- A completed deal is immediately followed by the next deal at the next
  candle's open (optionally after `deal_cooldown_bars`), i.e. this only
  models an "asap" start condition — signal-based/manual start
  conditions aren't meaningful to backtest without a matching signal feed.
- Only one deal is open at a time (a single bot runs a single pair with
  at most one concurrent deal).

ANTI-REPAINTING / no-look-ahead guarantee: within a single candle, OHLC
data can't tell you whether the high or the low happened first, so
run_backtest resolves every bar in a FIXED, PESSIMISTIC order — the
event that hurts the open position is always evaluated before the event
that helps it: liquidation check -> stop-loss/trailing check -> safety-
order fills (adverse-direction extreme) -> take-profit check (favorable-
direction extreme), every single bar, never the reverse and never
re-ordered based on how a config happens to score. A trade's recorded
outcome, once appended to a BacktestReport, is never mutated or
recomputed retroactively by a later pass — nothing in this module
"repaints" a result after the fact. This ordering can only ever
under-state a strategy's edge relative to the (unknowable) true
intra-bar path, never over-state it.

PERIOD POLICY (see ../../TESTING_POLICY.md — do not change this without
updating that file too): every backtest, grid search, walk-forward window,
and robust-search window is capped at MAX_BACKTEST_DAYS (14). This is a
deliberate, user-mandated limit, enforced here in code (not just by
convention) so a future caller can't silently drift to a longer window —
run_backtest raises ValueError if the candles it's given span more than
that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional

from symbot_python.exchange.utils import round_to_step
from symbot_python.strategy import dca_math
from symbot_python.strategy.stop_loss import StopLossInput, evaluate as evaluate_stop_loss

MAX_BACKTEST_DAYS = 14
_MAX_BACKTEST_SPAN_MS = (MAX_BACKTEST_DAYS + 0.5) * 86_400_000  # small buffer for candle-open-vs-close measurement rounding


def _validate_period(candles: list[list[float]], context: str = "backtest") -> None:
    if len(candles) < 2:
        return
    span_ms = candles[-1][0] - candles[0][0]
    if span_ms > _MAX_BACKTEST_SPAN_MS:
        span_days = span_ms / 86_400_000
        raise ValueError(
            f"{context}: candle span is {span_days:.1f} days, which exceeds the "
            f"hard {MAX_BACKTEST_DAYS}-day limit (see TESTING_POLICY.md). "
            f"Fetch fewer candles or a shorter interval instead of extending the period."
        )


ExitReason = Literal[
    "take_profit", "stop_loss", "trailing_stop", "liquidated", "reverse_drawdown", "optimizer_pivot", "end_of_data",
    "omlx_bailout",
]


@dataclass
class SafetyOrderGateContext:
    """Passed to run_backtest's optional safety_order_gate right before a
    triggered safety-order rung would fill, so an external decision
    system (OMLX) can veto it — bail out of the deal instead of
    averaging down. Purely additive: run_backtest's existing callers
    (walk_forward, grid_search, robust_search, the optimizer, every
    existing test) never pass this, so their behavior is unchanged."""
    bar_index: int
    bar_timestamp: float
    rung_index: int  # 1-based: which safety-order rung is about to fill
    current_average: float
    side: Literal["long", "short"]


def select_tier_maintenance_margin_rate(
    notional: float,
    risk_tiers: Optional[list[tuple[float, float, float]]],
    fallback_mmr: float,
) -> float:
    """Pick the maintenance margin rate for the tier `notional` actually
    falls into. risk_tiers is a list of (risk_limit_value, max_leverage,
    maintenance_margin_rate) sorted ascending by risk_limit_value, e.g.
    from Bybit's get_risk_limit (see exchange/risk_limits.py). Real
    exchanges do NOT use a single fixed rate — the rate gets worse
    (higher) as position notional crosses each tier's cap, which matters
    for a strategy specifically designed to compound into a large
    position. Falls back to a single fixed rate if no tier table is
    supplied (e.g. in tests, or when the tier lookup wasn't fetched).
    """
    if not risk_tiers:
        return fallback_mmr
    for risk_limit_value, _max_leverage, mmr in risk_tiers:
        if notional <= risk_limit_value:
            return mmr
    return risk_tiers[-1][2]  # notional exceeds every known tier: use the worst (last) one


@dataclass
class BacktestConfig:
    first_order_amount: float
    dca_order_amount: float
    dca_max_order: int
    dca_order_size_multiplier: float
    dca_order_start_distance: float
    dca_order_step_percent: float
    dca_order_step_percent_multiplier: float
    dca_take_profit_percent: float
    exchange_fee: float  # one-leg %, matches bot config's exchangeFee
    side: Literal["long", "short"] = "long"
    price_tick: float = 0.01
    min_move_amount: float = 0.0001
    deal_cooldown_bars: int = 0
    max_deals: Optional[int] = None

    dca_stop_loss_enabled: bool = False
    dca_stop_loss_percent: float = 0.0
    dca_stop_loss_reference: str = "average"
    dca_stop_loss_move_breakeven: bool = False
    dca_stop_loss_breakeven_trigger: float = 0.0

    dca_trailing_stop_enabled: bool = False
    dca_trailing_stop_distance: float = 0.0
    dca_trailing_activate_profit: float = 0.0
    dca_trailing_replaces_take_profit: bool = True

    # Stop-and-reverse — same fields and semantics as
    # strategy/models.py's BotConfig, so a searched/backtested value
    # means the same thing the live/paper engine will actually run with.
    # Checked independently of dca_stop_loss_enabled, every bar, against
    # the CURRENT deal's own drawdown (raw price-move %, same convention
    # as dca_stop_loss_percent). None/0 = disabled (default).
    reverse_drawdown_percent: Optional[float] = None
    reverse_cooldown_sec: float = 3600.0
    max_consecutive_reversals: int = 2

    # Leverage: "same margin, N x bigger position" convention — the
    # amounts above are MARGIN, actual position notional = margin * leverage.
    # leverage=1 behaves exactly like spot (no liquidation risk).
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.005
    # Full tier table from Bybit's get_risk_limit: [(risk_limit_value,
    # max_leverage, maintenance_margin_rate), ...] ascending by notional.
    # When supplied, the liquidation check looks up the CORRECT tier for
    # the position's current notional every tick instead of always using
    # maintenance_margin_rate — real maintenance margin gets worse as
    # notional grows past each tier's cap, which matters once
    # auto_size_to_funds has compounded a position to real size.
    risk_tiers: Optional[list[tuple[float, float, float]]] = None

    # Position sizing: when enabled, first_order_amount/dca_order_amount
    # are rescaled (preserving their configured ratio/growth curve) at
    # the start of EVERY deal so that calculate_max_funds(...) — the
    # margin required if the entire ladder fills — always equals
    # funds_utilization_percent of the CURRENT equity. This is what
    # makes position size compound with account growth/shrinkage rather
    # than staying fixed at whatever was typed into the form.
    auto_size_to_funds: bool = True
    funds_utilization_percent: float = 98.0

    # Funding: perpetual futures charge/pay funding periodically (Bybit:
    # every 8h) based on position NOTIONAL — a real, recurring cost spot
    # DCA never has. funding_events is a list of (timestamp_ms, rate)
    # pairs, sorted ascending, e.g. from Bybit's get_funding_rate_history
    # (rate is the raw fraction, e.g. 0.0001 for 0.01%). A LONG pays when
    # rate > 0 (and receives when rate < 0); a SHORT is the mirror image.
    # Applied against whatever notional is actually open at the moment
    # each funding timestamp is crossed. Empty by default so existing
    # results that never supplied this are unaffected.
    funding_events: list[tuple[float, float]] = field(default_factory=list)


# "Is this a real win" thresholds, ported from the user's live ETH bot
# (unified-combo-grid's eth_trader_bt.py WIN_FEE_MULT/MIN_WIN_PRICE_PCT):
# a trade that merely edges out what it paid in fees isn't a real edge,
# it's noise. A trade failing this still counts as a trade and its actual
# profit_quote still flows into total_profit_quote/equity — this only
# reclassifies it out of the "win" bucket for win_rate reporting.
WIN_FEE_MULT = 2.0
# Take profit is a product policy, not an optimization dimension. Keep one
# authoritative value: percent-NUMBER (0.33 means "0.33%").
FIXED_TAKE_PROFIT_PERCENT = 0.33
MIN_WIN_PRICE_PCT = FIXED_TAKE_PROFIT_PERCENT  # matching
# raw_move_percent's own convention below — NOT the fraction (0.0033) the
# reference eth_trader_bt.py uses, which compares against its own raw_pct
# (a fraction, e.g. 0.0033 for a 0.33% move). This codebase's raw_move_percent
# is `raw_move / average * 100`, a percent-number like every other
# "_percent" field here (e.g. dca_take_profit_percent=1.5 means 1.5%) — so
# the threshold must be in that same convention. Comparing a percent-number
# against a fraction directly (0.0033) made this floor a near no-op in
# production: a 0.10% move with zero fees would classify as a real win,
# far under the documented 0.33% intent. Found and fixed after an
# independent audit of the live/paper engine (dca_bot.py) traced the exact
# same threshold and caught the unit mismatch.


def clamp_take_profit_percent(value: float) -> float:
    """Keep the generic simulator's historical safety floor.

    The production tester sets its configuration to
    ``FIXED_TAKE_PROFIT_PERCENT`` and omits TP from its search grid; this
    helper remains a floor so isolated simulator tests can model other
    strategies without weakening the production policy.
    """
    return max(MIN_WIN_PRICE_PCT, value)


@dataclass
class BacktestTrade:
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    average: float
    qty: float
    profit_quote: float
    profit_percent: float
    safety_orders_used: int
    exit_reason: ExitReason
    raw_move_percent: float = 0.0  # price move %, before fees/funding — used for the tiny-win filter
    estimated_fees_quote: float = 0.0  # approx round-trip fee in dollar terms, same purpose
    funding_cost_quote: float = 0.0  # cumulative funding paid (positive) or received (negative) this deal

    side: Literal["long", "short"] = "long"
    leverage: float = 1.0

    def is_real_win(self, win_fee_mult: float = WIN_FEE_MULT, min_win_price_pct: float = MIN_WIN_PRICE_PCT) -> bool:
        if self.profit_quote <= 0:
            return False
        if self.raw_move_percent < min_win_price_pct:
            return False
        if self.estimated_fees_quote > 0 and self.profit_quote <= win_fee_mult * self.estimated_fees_quote:
            return False
        return True


@dataclass
class BacktestReport:
    trades: list[BacktestTrade] = field(default_factory=list)
    starting_equity: float = 0.0
    final_equity: float = 0.0
    max_drawdown_quote: float = 0.0
    max_drawdown_percent: float = 0.0
    period_days: float = 0.0  # actual candle-span of the window tested, for CAGR/Sharpe annualization

    @property
    def total_profit_percent(self) -> float:
        return (self.total_profit_quote / self.starting_equity * 100) if self.starting_equity else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Per-TRADE return Sharpe (not per-bar like some references —
        this backtest doesn't track a bar-by-bar mark-to-market equity
        curve, only realized P&L at each trade's close), annualized by
        how many trades/year this window's trade frequency implies.
        0.0 with fewer than 2 trades or zero variance (nothing to
        annualize/divide by).
        """
        import statistics

        if len(self.trades) < 2 or self.period_days <= 0:
            return 0.0
        returns = [t.profit_percent / 100 for t in self.trades]
        stdev = statistics.stdev(returns)
        if stdev == 0:
            return 0.0
        trades_per_year = len(self.trades) / self.period_days * 365
        return statistics.mean(returns) / stdev * (trades_per_year ** 0.5)

    @property
    def cagr_percent(self) -> float:
        """Annualized return, extrapolated from this window's actual
        return over its actual span — clamped to +-1e8% since a short
        window's return compounded out to a full year can blow up to an
        absurd number (this is informational only, never used to rank
        or select a config — a short backtest's CAGR is not a serious
        forecast of annual performance, just a normalized-for-comparison
        figure).
        """
        if self.starting_equity <= 0 or self.period_days <= 0:
            return 0.0
        years = self.period_days / 365
        ratio = self.final_equity / self.starting_equity
        if ratio <= 0:
            return -100.0  # total loss (or worse) -> can't take a fractional power of a non-positive base
        try:
            cagr = (ratio ** (1 / years) - 1) * 100
        except (OverflowError, ValueError):
            return 1e8
        if cagr != cagr or cagr in (float("inf"), float("-inf")):  # NaN check without importing math
            return 1e8
        return max(-100.0, min(cagr, 1e8))

    @property
    def total_profit_quote(self) -> float:
        return self.final_equity - self.starting_equity

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.is_real_win())

    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.trades if not t.is_real_win())

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.win_count / len(self.trades)

    @property
    def average_profit_percent(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.profit_percent for t in self.trades) / len(self.trades)

    # -- gross profit/loss, profit factor -----------------------------------

    @property
    def gross_profit_quote(self) -> float:
        """Sum of profit_quote over REAL wins only — matches the reference
        eth_trader_bt.py exactly: `if <passes win filter>: gw += part_pnl
        else: gl += abs(part_pnl)`. A trade that is "not a real win" (raw
        loss OR a positive-but-too-tiny-to-count win) goes to the loss
        side, full stop — a trade can never be invisible to both sides,
        which is what let a loss streak show up with no matching loss
        amount before this fix (max_loss_streak used is_real_win() while
        gross/max_loss_quote used raw profit_quote sign — two different
        definitions of "loss" that could disagree on the same trade).
        """
        return sum(t.profit_quote for t in self.trades if t.is_real_win())

    @property
    def gross_loss_quote(self) -> float:
        """Sum of |profit_quote| (as a negative total) over every trade
        that is NOT a real win — including a tiny/filtered win, exactly
        like the reference bot's `gl += abs(part_pnl)` in its else branch.
        """
        return -sum(abs(t.profit_quote) for t in self.trades if not t.is_real_win())

    @property
    def profit_factor(self) -> float:
        """gross_profit / abs(gross_loss) — the standard trading-system
        metric: >1 means the system makes more than it loses. inf if
        there were wins and zero losses; 0.0 if there were no wins at all.
        """
        gross_loss = abs(self.gross_loss_quote)
        if gross_loss == 0:
            return float("inf") if self.gross_profit_quote > 0 else 0.0
        return self.gross_profit_quote / gross_loss

    # -- max/average win and loss --------------------------------------------

    @property
    def max_win_quote(self) -> float:
        wins = [t.profit_quote for t in self.trades if t.is_real_win()]
        return max(wins, default=0.0)

    @property
    def max_loss_quote(self) -> float:
        losses = [-abs(t.profit_quote) for t in self.trades if not t.is_real_win()]
        return min(losses, default=0.0)

    @property
    def max_win_percent(self) -> float:
        wins = [t.profit_percent for t in self.trades if t.is_real_win()]
        return max(wins, default=0.0)

    @property
    def max_loss_percent(self) -> float:
        losses = [-abs(t.profit_percent) for t in self.trades if not t.is_real_win()]
        return min(losses, default=0.0)

    @property
    def average_win_quote(self) -> float:
        wins = [t.profit_quote for t in self.trades if t.is_real_win()]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def average_loss_quote(self) -> float:
        losses = [abs(t.profit_quote) for t in self.trades if not t.is_real_win()]
        return -(sum(losses) / len(losses)) if losses else 0.0

    @property
    def average_win_percent(self) -> float:
        wins = [t.profit_percent for t in self.trades if t.is_real_win()]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def average_loss_percent(self) -> float:
        losses = [abs(t.profit_percent) for t in self.trades if not t.is_real_win()]
        return -(sum(losses) / len(losses)) if losses else 0.0

    # -- streaks --------------------------------------------------------------

    @property
    def max_win_streak(self) -> int:
        return self._max_streak(is_win=True)

    @property
    def max_loss_streak(self) -> int:
        return self._max_streak(is_win=False)

    def _max_streak(self, is_win: bool) -> int:
        best = current = 0
        for t in self.trades:
            if t.is_real_win() == is_win:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    # -- costs (transparency on what ate into profit) ---------------------------

    @property
    def total_fees_quote(self) -> float:
        return sum(t.estimated_fees_quote for t in self.trades)

    @property
    def total_funding_quote(self) -> float:
        return sum(t.funding_cost_quote for t in self.trades)

    # -- other useful aggregates -------------------------------------------------

    @property
    def liquidation_count(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "liquidated")

    @property
    def average_trade_duration_hours(self) -> float:
        if not self.trades:
            return 0.0
        return sum((t.exit_ts - t.entry_ts) for t in self.trades) / len(self.trades) / 3_600_000

    @property
    def average_safety_orders_used(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.safety_orders_used for t in self.trades) / len(self.trades)


def _identity_filter_price(config: BacktestConfig):
    def f(price: float) -> float:
        return round_to_step(price, config.price_tick) if config.price_tick else price

    return f


def _identity_filter_amount(config: BacktestConfig):
    def f(qty: float) -> float:
        return round_to_step(qty, config.min_move_amount) if config.min_move_amount else qty

    return f


def build_ladder(config: BacktestConfig, entry_price: float) -> list[dca_math.OrderRung]:
    """Compute the full static order ladder for a deal starting at
    entry_price, computed once at deal creation rather than recomputed
    as the market moves.

    first_order_amount/dca_order_amount are MARGIN. When config.leverage
    > 1, each rung's actual notional (what qty/amount/average/target are
    computed from) is margin * leverage — "same margin, N x bigger
    position." At leverage=1 this is a no-op (notional == margin).

    LONG: safety orders sit BELOW entry (price falling triggers them),
    target sits ABOVE average. SHORT is the mirror image: safety orders
    sit ABOVE entry, target sits BELOW average — same deviation
    percentages, opposite direction.
    """
    filter_price = _identity_filter_price(config)
    filter_amount = _identity_filter_amount(config)
    leverage = config.leverage if config.leverage > 0 else 1.0
    is_short = config.side == "short"
    direction = 1 if is_short else -1  # sign applied to the deviation %

    orders: list[dca_math.OrderRung] = []

    base_notional = config.first_order_amount * leverage
    base_qty_raw = base_notional / entry_price
    base_adj = dca_math.calculate_adjustments(
        entry_price, base_qty_raw, config.exchange_fee, config.min_move_amount, filter_amount, filter_price
    )
    orders.append(
        dca_math.OrderRung(
            price=entry_price, qty=base_adj.order_qty, amount=base_adj.order_amount,
            qty_sum=0, sum=0, average=0, target=0, filled=1,
        )
    )

    prev_margin = config.dca_order_amount
    for i in range(1, config.dca_max_order + 1):
        if i == 1:
            price = filter_price(entry_price * (1 + direction * config.dca_order_start_distance / 100))
            margin = config.dca_order_amount
        else:
            deviation = dca_math.get_deviation_dca(
                config.dca_order_step_percent, config.dca_order_step_percent_multiplier, i
            )
            price = filter_price(entry_price * (1 + direction * deviation / 100))
            margin = prev_margin * config.dca_order_size_multiplier
        notional = margin * leverage
        qty_raw = notional / price
        adj = dca_math.calculate_adjustments(
            price, qty_raw, config.exchange_fee, config.min_move_amount, filter_amount, filter_price
        )
        orders.append(
            dca_math.OrderRung(
                price=price, qty=adj.order_qty, amount=adj.order_amount,
                qty_sum=0, sum=0, average=0, target=0, filled=0,
            )
        )
        prev_margin = margin

    return dca_math.recalculate_orders(
        orders, None, config.exchange_fee, config.min_move_amount,
        config.dca_take_profit_percent, filter_amount, filter_price, config.price_tick,
        side=config.side,
    )


def run_backtest(
    config: BacktestConfig, candles: list[list[float]], starting_equity: float,
    *, config_updates: Optional[list[tuple[int, BacktestConfig]]] = None,
    close_at_end: bool = True,
    safety_order_gate: Optional[Callable[[SafetyOrderGateContext], bool]] = None,
) -> BacktestReport:
    """Internal candle simulator used by walk-forward training/evaluation.

    Config updates take effect at a candle's OPEN, using only a winner
    selected before that candle. Open deals retain their own configuration.
    An optimizer direction change executes a pivot; same-side refreshes only
    affect future deals. Intermediate walk-forward prefixes mark open equity
    without creating an end-of-window exit or charging a hypothetical exit fee.
    """
    _validate_period(candles, context="run_backtest")
    # Non-negotiable floor, enforced here rather than only at the
    # SEARCH_GRID definition — every backtest/grid_search/random_search/
    # walk_forward call funnels through this one function, so clamping
    # here structurally covers all of them, including a stale
    # param-library row recorded before this floor existed.
    config = replace(config, dca_take_profit_percent=clamp_take_profit_percent(config.dca_take_profit_percent))
    period_days = (candles[-1][0] - candles[0][0]) / 86_400_000 if len(candles) >= 2 else 0.0
    report = BacktestReport(starting_equity=starting_equity, final_equity=starting_equity, period_days=period_days)
    if not candles:
        return report

    updates = list(config_updates or [])
    if any(index < 0 or index >= len(candles) for index, _ in updates):
        raise ValueError("config update index must refer to a supplied candle")
    if any(a[0] >= b[0] for a, b in zip(updates, updates[1:])):
        raise ValueError("config updates must have strictly increasing candle indices")
    update_index = 0

    def apply_updates(bar_index):
        nonlocal config, update_index
        changed_side = False
        while update_index < len(updates) and updates[update_index][0] <= bar_index:
            _, new_config = updates[update_index]
            changed_side = changed_side or new_config.side != config.side
            config = replace(new_config, dca_take_profit_percent=clamp_take_profit_percent(new_config.dca_take_profit_percent))
            update_index += 1
        return changed_side

    peak_equity = starting_equity
    equity = starting_equity
    i = 0
    n = len(candles)

    # Stop-and-reverse state, persisted ACROSS outer-loop iterations
    # (unlike everything else above, which is per-deal). current_side
    # flips to the opposite of config.side for exactly the ONE deal
    # that immediately follows a reverse_drawdown exit, then reverts to
    # config.side again — a normal (non-reversal) close always reverts,
    # matching dca_bot_manager.py's live/paper behavior exactly (a
    # flipped deal's own BotConfig copy is never written back to the
    # bot's canonical, un-flipped config). pending_entry overrides the
    # next deal's entry (ts, price) to the exact reversal fill instead
    # of the next bar's open — the reversal is one atomic action, not a
    # close now and a separately-timed open later.
    current_side = config.side
    consecutive_reversals = 0
    last_reversal_ts: Optional[float] = None
    pending_entry: Optional[tuple[float, float]] = None
    pending_config: Optional[BacktestConfig] = None
    pending_qty: Optional[float] = None
    pending_at_open = False

    while i < n:
        apply_updates(i)
        if pending_entry is None:
            current_side = config.side
        if config.max_deals is not None and len(report.trades) >= config.max_deals:
            break

        is_short = current_side == "short"
        deal_config = pending_config or replace(config, side=current_side)
        if pending_config is None and config.auto_size_to_funds:
            target_budget = max(equity, 0.0) * (config.funds_utilization_percent / 100)
            sized_first, sized_dca = dca_math.solve_order_sizing_for_budget(
                deal_config.first_order_amount, deal_config.dca_order_amount, deal_config.dca_max_order,
                deal_config.dca_order_size_multiplier, deal_config.exchange_fee, target_budget,
            )
            deal_config = replace(deal_config, first_order_amount=sized_first, dca_order_amount=sized_dca)

        leverage = deal_config.leverage if deal_config.leverage > 0 else 1.0

        if pending_entry is not None:
            entry_ts, entry_price = pending_entry
            pending_entry = None
        else:
            entry_ts, entry_price = candles[i][0], candles[i][1]  # enter at this bar's open
        orders = build_ladder(deal_config, entry_price)
        is_pivot = pending_qty is not None
        if is_pivot:
            # The atomic reversal already filled this exact quantity.
            # Rebuilding the safety ladder must not gross up its base again.
            fp = _identity_filter_price(deal_config)
            orders[0] = replace(
                orders[0], qty=pending_qty, amount=pending_qty * entry_price,
                qty_sum=pending_qty, sum=pending_qty * entry_price, average=entry_price,
                target=dca_math.calculate_target_price(
                    entry_price, deal_config.dca_take_profit_percent, deal_config.exchange_fee,
                    fp, deal_config.price_tick, side=current_side), filled=1, manual=True,
            )
            orders = dca_math.recalculate_orders(
                orders, None, deal_config.exchange_fee, deal_config.min_move_amount,
                deal_config.dca_take_profit_percent, _identity_filter_amount(deal_config),
                fp, deal_config.price_tick, side=current_side,
            )
        pending_config = None
        pending_qty = None
        filled_count = 1
        # trail_high_price is the trailing reference EXTREME regardless of
        # side: highest price for long, lowest for short (see stop_loss.py).
        trail_high_price = entry_price if is_pivot else (candles[i][3] if is_short else candles[i][2])
        breakeven_armed = False
        active_stop_loss_price = 0.0

        exit_reason: Optional[ExitReason] = None
        exit_price = 0.0
        exit_ts = entry_ts
        funding_cost_accum = 0.0
        funding_idx = 0
        while (
            funding_idx < len(deal_config.funding_events)
            and deal_config.funding_events[funding_idx][0] <= entry_ts
        ):
            funding_idx += 1  # skip any funding events that already happened before this deal opened

        j = i if pending_at_open else i + 1
        pending_at_open = False
        while j < n:
            ts, o, h, l, c, v = candles[j]
            last_filled = orders[filled_count - 1]
            direction_changed = apply_updates(j)
            if direction_changed and config.side != current_side:
                # At a boundary only the opening price is known. Do not
                # use this candle's future high/low/close to fill the pivot.
                while (funding_idx < len(deal_config.funding_events)
                       and deal_config.funding_events[funding_idx][0] <= ts):
                    funding_cost_accum += last_filled.qty_sum * o * deal_config.funding_events[funding_idx][1] * (-1 if is_short else 1)
                    funding_idx += 1
                mmr = select_tier_maintenance_margin_rate(
                    last_filled.qty_sum * o, deal_config.risk_tiers, deal_config.maintenance_margin_rate)
                liq = dca_math.calculate_liquidation_price(last_filled.average, leverage, mmr, side=current_side)
                hit_liquidation = leverage > 1 and (o >= liq if is_short else o <= liq)
                exit_reason = "liquidated" if hit_liquidation else "optimizer_pivot"
                exit_price = liq if hit_liquidation else o
                exit_ts = ts
                break
            trail_high_price = min(trail_high_price, l) if is_short else max(trail_high_price, h)

            # Funding: apply every real funding event crossed while this
            # position is open, against whatever notional is open right
            # now (mark-to-market at this bar's close). A long PAYS when
            # rate > 0; a short is the exact mirror.
            while (
                funding_idx < len(deal_config.funding_events)
                and deal_config.funding_events[funding_idx][0] <= ts
            ):
                rate = deal_config.funding_events[funding_idx][1]
                current_notional = last_filled.qty_sum * c
                funding_cost_accum += current_notional * rate * (-1.0 if is_short else 1.0)
                funding_idx += 1

            if leverage > 1 and last_filled.average:
                current_notional_for_tier = last_filled.qty_sum * c
                effective_mmr = select_tier_maintenance_margin_rate(
                    current_notional_for_tier, deal_config.risk_tiers, deal_config.maintenance_margin_rate,
                )
                liq_price = dca_math.calculate_liquidation_price(
                    last_filled.average, leverage, effective_mmr, side=deal_config.side,
                )
                liquidated = h >= liq_price if is_short else l <= liq_price
                if liquidated:
                    exit_reason = "liquidated"
                    exit_price = liq_price
                    exit_ts = ts
                    break

            # Worst case within the bar: price rising hurts a short,
            # price falling hurts a long. Computed unconditionally
            # (cheap) so both the stop-loss evaluation below AND the
            # reverse-drawdown check can share it — reverse_drawdown_percent
            # is checked independently of dca_stop_loss_enabled, exactly
            # like the live/paper engine (dca_bot.py).
            worst_case_price = h if is_short else l
            profit_pct = (
                (
                    (last_filled.average - c) / last_filled.average * 100
                    if is_short
                    else (c - last_filled.average) / last_filled.average * 100
                )
                - deal_config.exchange_fee
                if last_filled.average
                else 0.0
            )

            if deal_config.dca_stop_loss_enabled or deal_config.dca_trailing_stop_enabled:
                sl_result = evaluate_stop_loss(
                    StopLossInput(
                        enabled=deal_config.dca_stop_loss_enabled,
                        price=worst_case_price,
                        average=last_filled.average,
                        stop_loss_percent=deal_config.dca_stop_loss_percent,
                        reference=deal_config.dca_stop_loss_reference,  # type: ignore[arg-type]
                        last_safety_order_price=orders[filled_count - 1].price,
                        fee_rate=deal_config.exchange_fee,
                        move_breakeven=deal_config.dca_stop_loss_move_breakeven,
                        breakeven_trigger=deal_config.dca_stop_loss_breakeven_trigger,
                        profit_percentage=profit_pct,
                        breakeven_armed=breakeven_armed,
                        active_stop_loss_price=active_stop_loss_price,
                        trailing_enabled=deal_config.dca_trailing_stop_enabled,
                        trailing_distance=deal_config.dca_trailing_stop_distance,
                        trailing_activate_profit=deal_config.dca_trailing_activate_profit,
                        trail_high_price=trail_high_price,
                        side=deal_config.side,
                    )
                )
                if sl_result.breakeven_armed:
                    breakeven_armed = True
                ratchet_improves = (
                    active_stop_loss_price == 0.0 or (
                        sl_result.level < active_stop_loss_price if is_short
                        else sl_result.level > active_stop_loss_price
                    )
                )
                if ratchet_improves and (sl_result.breakeven_armed or sl_result.trailing_active):
                    active_stop_loss_price = sl_result.level
                if sl_result.triggered:
                    exit_reason = "trailing_stop" if sl_result.hit_label == "trailing" else "stop_loss"
                    exit_price = sl_result.level
                    exit_ts = ts
                    break

            # Stop-and-reverse: checked only when stop-loss isn't already
            # firing this bar (a plain hard stop-loss, if configured, is
            # the more fundamental protection and always wins), and only
            # once the position is at least reverse_drawdown_percent
            # underwater. Hysteresis mirrors dca_bot.py's
            # _evaluate_reverse_drawdown exactly, using simulated time
            # (candle timestamps, milliseconds) instead of wall-clock time.
            if deal_config.reverse_drawdown_percent and profit_pct <= -deal_config.reverse_drawdown_percent:
                cooldown_ok = (
                    last_reversal_ts is None
                    or (ts - last_reversal_ts) / 1000 >= deal_config.reverse_cooldown_sec
                )
                cap_ok = consecutive_reversals < deal_config.max_consecutive_reversals
                if cooldown_ok and cap_ok:
                    exit_reason = "reverse_drawdown"
                    exit_price = worst_case_price
                    exit_ts = ts
                    break

            bailed_out = False
            for idx in range(filled_count, len(orders)):
                rung_triggered = h >= orders[idx].price if is_short else l <= orders[idx].price
                if not rung_triggered:
                    break
                if safety_order_gate is not None and not safety_order_gate(
                    SafetyOrderGateContext(
                        bar_index=j, bar_timestamp=ts, rung_index=idx,
                        current_average=orders[filled_count - 1].average,
                        side=deal_config.side,
                    )
                ):
                    # Vetoed: bail out of the deal now instead of averaging
                    # down into this rung — exit at this bar's worst-case
                    # price, same pessimistic convention as the
                    # stop-loss/reverse-drawdown exits just above.
                    exit_reason = "omlx_bailout"
                    exit_price = worst_case_price
                    exit_ts = ts
                    bailed_out = True
                    break
                orders[idx] = replace(orders[idx], filled=1)
                filled_count = idx + 1
            if bailed_out:
                break

            current_average = orders[filled_count - 1].average
            current_profit_pct = (
                (
                    (current_average - c) / current_average * 100
                    if is_short
                    else (c - current_average) / current_average * 100
                )
                if current_average
                else None
            )
            trailing_active = (
                deal_config.dca_trailing_stop_enabled
                and deal_config.dca_trailing_activate_profit is not None
                and trail_high_price > 0
                and current_profit_pct is not None
                and current_profit_pct >= deal_config.dca_trailing_activate_profit
            )
            suppress_take_profit = trailing_active and deal_config.dca_trailing_replaces_take_profit
            target = orders[filled_count - 1].target
            take_profit_hit = l <= target if is_short else h >= target
            if not suppress_take_profit and take_profit_hit:
                exit_reason = "take_profit"
                exit_price = target
                exit_ts = ts
                break

            j += 1

        last_filled = orders[filled_count - 1]
        if exit_reason is None and not close_at_end:
            mark = candles[-1][4]
            raw = last_filled.qty_sum * ((last_filled.average - mark) if is_short else (mark - last_filled.average))
            entry_fees = last_filled.sum * deal_config.exchange_fee / 100
            report.final_equity = equity + raw - entry_fees - funding_cost_accum
            return report
        if exit_reason is None:
            # Ran out of data with the deal still open: mark-to-market
            # close at the final candle's close, for reporting purposes.
            exit_reason = "end_of_data"
            exit_price = candles[-1][4]
            exit_ts = candles[-1][0]

        # Note: profit_quote is the real dollar P&L on the (possibly
        # leverage-amplified) notional position — qty_sum already
        # reflects leverage from build_ladder, so this needs no special
        # case for a "liquidated" exit: plugging in the liquidation
        # price here naturally yields a loss of approximately the full
        # margin committed. profit_percent is the ROI on MARGIN actually
        # committed (last_filled.sum is cumulative NOTIONAL, so dividing
        # by leverage recovers margin) — NOT the raw price move. A $35
        # profit on a 0.36% price move only makes sense once you see it's
        # leveraged; reporting the raw price-move % next to a
        # leverage-amplified dollar figure looked like a bug because it
        # was one — the two numbers must be on the same (margin) basis to
        # be read together. SHORT mirrors the sign: profit comes from
        # exit BELOW average.
        raw_move = (
            (last_filled.average - exit_price) if is_short else (exit_price - last_filled.average)
        )
        entry_fees = last_filled.sum * deal_config.exchange_fee / 100
        profit_quote = last_filled.qty_sum * raw_move - entry_fees - (
            last_filled.qty_sum * exit_price * (deal_config.exchange_fee / 100)
        ) - funding_cost_accum
        margin_committed = (last_filled.sum / leverage) if leverage else last_filled.sum
        profit_percent = (profit_quote / margin_committed * 100) if margin_committed else 0.0
        # Raw price move %, BEFORE fees/funding — the tiny-win filter looks
        # at this, not profit_percent, since a trade that only "won" by a
        # hair after fees shouldn't count even if the raw move was decent.
        raw_move_percent = (raw_move / last_filled.average * 100) if last_filled.average else 0.0
        # Quantity gross-up increases exposure; it does not pay commissions.
        estimated_fees_quote = entry_fees + last_filled.qty_sum * exit_price * deal_config.exchange_fee / 100

        reverse_qty = 0.0
        if exit_reason in {"reverse_drawdown", "optimizer_pivot"}:
            # Match the paper engine: size from FREE CASH before closing,
            # excluding locked margin, unpaid funding and unrealised P/L.
            available = max(0.0, equity - margin_committed - entry_fees)
            first_margin = deal_config.first_order_amount
            if deal_config.auto_size_to_funds:
                first_margin, _ = dca_math.solve_order_sizing_for_budget(
                    deal_config.first_order_amount, deal_config.dca_order_amount,
                    deal_config.dca_max_order, deal_config.dca_order_size_multiplier,
                    deal_config.exchange_fee, available * deal_config.funds_utilization_percent / 100,
                )
            reverse_qty = _identity_filter_amount(deal_config)(first_margin * leverage / exit_price)
            # Same close-flat fallback when the extra flip margin cannot fit.
            after_close = equity + profit_quote + funding_cost_accum
            required = reverse_qty * exit_price * (1 / leverage + deal_config.exchange_fee / 100)
            if required > after_close or reverse_qty <= 0:
                reverse_qty = 0.0

        report.trades.append(
            BacktestTrade(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry_price=entry_price,
                exit_price=exit_price,
                average=last_filled.average,
                qty=last_filled.qty_sum,
                profit_quote=profit_quote,
                profit_percent=profit_percent,
                safety_orders_used=filled_count - 1,
                exit_reason=exit_reason,
                raw_move_percent=raw_move_percent,
                estimated_fees_quote=estimated_fees_quote,
                funding_cost_quote=funding_cost_accum,
                side=current_side, leverage=leverage,
            )
        )

        equity += profit_quote
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        report.max_drawdown_quote = max(report.max_drawdown_quote, drawdown)
        if peak_equity > 0:
            report.max_drawdown_percent = max(
                report.max_drawdown_percent, drawdown / peak_equity * 100
            )

        if exit_reason == "end_of_data":
            break

        if equity <= 0:
            # Bankrupt — a real account can't open another deal with no
            # equity left. Without this, target_budget floors to 0 next
            # iteration (see auto_size_to_funds above),
            # solve_order_sizing_for_budget returns a degenerate
            # (0.0, 0.0) sizing, and the loop kept "opening" zero-qty/
            # zero-profit phantom deals for the rest of the dataset —
            # inflating this candidate's trade count/win_rate with
            # no-op trades instead of just stopping. Doesn't change
            # total_profit_quote or max_drawdown (already correct at
            # the point of going bankrupt), only removes trailing noise.
            break

        # Streak bookkeeping, mirroring dca_bot.py's _handle_sell: any
        # reversal-triggered close extends the streak; literally any
        # other close reason means the strategy resolved normally —
        # reset it, so a later, unrelated drawdown starts a fresh count.
        if exit_reason == "reverse_drawdown":
            consecutive_reversals += 1
            last_reversal_ts = exit_ts
        else:
            consecutive_reversals = 0
        if reverse_qty > 0:
            current_side = "short" if current_side == "long" else "long"
            implied_margin = reverse_qty * exit_price / leverage
            scale = implied_margin / deal_config.first_order_amount if deal_config.first_order_amount else 1.0
            pending_config = replace(
                deal_config, side=current_side, first_order_amount=implied_margin,
                dca_order_amount=deal_config.dca_order_amount * scale,
            )
            pending_qty = reverse_qty
            pending_entry = (exit_ts, exit_price)
            pending_at_open = exit_reason == "optimizer_pivot"
            i = j
        else:
            current_side = config.side
            i = j + 1 + config.deal_cooldown_bars

    report.final_equity = equity
    return report
