# Testing Period Policy — READ BEFORE CHANGING ANY BACKTEST/OPTIMIZATION CODE

**Every backtest, grid search, walk-forward window, and robust-search
window is capped at 14 days. This is a hard, user-mandated rule. Do not
extend it, work around it, or add a way to bypass it — if a task seems
to need more than 14 days of data, stop and ask the user first instead
of quietly changing this policy.**

## Where this is enforced

- `symbot_python/strategy/backtest.py`: `MAX_BACKTEST_DAYS = 14` and
  `_validate_period()`, called at the top of `run_backtest()`. Raises
  `ValueError` if the candles passed in span more than ~14.5 days (the
  0.5-day buffer only accounts for candle-open-vs-close measurement
  rounding, e.g. 336 hourly candles spans 13.96 days between opens).
- `symbot_python/strategy/optimize.py`: the same check is called again
  at the top of `grid_search()`, `walk_forward()` (validated against the
  FULL input candle list, before it gets sliced into smaller in-sample/
  out-of-sample windows — a long history sliced into small windows must
  still be rejected), and `robust_search()` (validated against every
  window in the list).
- `symbot_python/signals/candles.py`: `DEFAULT_BACKTEST_DAYS = 14` and
  `bars_for_days()` are what every candle-fetching call site (the
  `optimizer` background task inside `run_everything.py`) uses to
  compute how many candles to request — this is what keeps normal usage
  inside the limit in the first place; the checks above are the backstop
  that makes it impossible to accidentally exceed it even so.

## Why

The user explicitly locked this in after a real finding: a "best"
config selected by testing against several 14-day windows (even ones
spread across many months, covering both up and down markets) still
looked completely safe — until it was run over one continuous 300-day
period and hit 4 liquidation events and a 98.55% max drawdown. Longer
continuous windows surface risk that no set of short windows reliably
catches, and the user's explicit call, given that, was to standardize
everything on strict 14-day testing rather than chase longer windows —
so results stay simple, fast, comparable across runs, and consistent
with how this project's numbers get discussed. Extending the period
without being asked would silently change what every past result in
this project means.

## If you (a future session, human or AI) think this needs to change

It might, someday — but that's the user's call, not something to infer
from a task. Ask first. Don't raise `MAX_BACKTEST_DAYS`, don't add a
`max_days` override parameter that defaults higher, don't fetch extra
data "just to check" outside these functions and hand-roll a longer
backtest around them.
