"""Single source of truth for the funds-utilization band — non-negotiable,
same policy class as the leverage band (see leverage_policy.py) and the
14-day backtest cap. funds_utilization_percent must never go below
MIN_FUNDS_UTILIZATION_PERCENT anywhere in this codebase: a deal sized
below this floor commits too little of the account to the trade for even
a full, successful ladder to produce a meaningful dollar profit — "DCA
needs to be smart enough to make a decent amount," not just technically
avoid losing. The continuous optimizer's search grid and paper trading's
winner-sync both import from here rather than hardcoding their own
bounds, so the floor can never drift out of sync between them.
"""

from __future__ import annotations

MIN_FUNDS_UTILIZATION_PERCENT = 25.0
MAX_FUNDS_UTILIZATION_PERCENT = 98.0
DEFAULT_FUNDS_UTILIZATION_PERCENT = 78.0


def clamp_funds_utilization(value: float) -> float:
    """Forces any funds_utilization_percent value into
    [MIN_FUNDS_UTILIZATION_PERCENT, MAX_FUNDS_UTILIZATION_PERCENT] —
    defense in depth for any path that accepts this value from a stored
    param-library row (which may predate this floor existing) or a
    fallback default, independent of whatever validation exists further
    up the call chain.
    """
    return max(MIN_FUNDS_UTILIZATION_PERCENT, min(MAX_FUNDS_UTILIZATION_PERCENT, value))
