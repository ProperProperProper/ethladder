"""Single source of truth for the leverage band — non-negotiable, same
policy class as the 14-day backtest cap (see TESTING_POLICY.md /
[[feedback_non_negotiables]]). Leverage must never exceed MAX_LEVERAGE
and never go below MIN_LEVERAGE, anywhere in this codebase: the manual
backtest form, paper trading, and the continuous optimizer's search
grid all import from here rather than hardcoding their own bounds, so
the band can never drift out of sync between them.
"""

from __future__ import annotations

MIN_LEVERAGE = 9.0
MAX_LEVERAGE = 11.0
DEFAULT_LEVERAGE = 11.0


def clamp_leverage(value: float) -> float:
    """Forces any leverage value into [MIN_LEVERAGE, MAX_LEVERAGE] —
    defense in depth for any path that accepts a leverage value from a
    form submission or a stored param-library row, independent of
    whatever validation exists further up the call chain.
    """
    return max(MIN_LEVERAGE, min(MAX_LEVERAGE, value))
