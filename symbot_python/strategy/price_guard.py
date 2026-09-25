"""Tick-price sanity guard.

Pure function: rejects implausible ticks (garbage/glitched price feed)
relative to the deal's own last-known-good average, without ever
requiring network/exchange access. Fails OPEN (treats price as plausible)
whenever it cannot make a confident judgement — the caller is expected to
also run a separate zero/negative-price check before trusting a "plausible"
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

DEFAULT_HIGH_RATIO = 2.0
DEFAULT_LOW_RATIO = 10.0

Reason = Literal["invalid_price", "no_reference", "above_band", "below_band", "ok"]


@dataclass
class PriceSanityResult:
    plausible: bool
    reason: Reason
    ratio: Optional[float] = None


def evaluate_price_sanity(
    price: float,
    reference: Optional[float],
    max_high_ratio: float = DEFAULT_HIGH_RATIO,
    max_low_ratio: float = DEFAULT_LOW_RATIO,
) -> PriceSanityResult:
    high_ratio = max_high_ratio if max_high_ratio and max_high_ratio > 1 else DEFAULT_HIGH_RATIO
    low_ratio = max_low_ratio if max_low_ratio and max_low_ratio > 1 else DEFAULT_LOW_RATIO

    if price is None or price <= 0:
        return PriceSanityResult(plausible=True, reason="invalid_price")

    if not reference or reference <= 0:
        return PriceSanityResult(plausible=True, reason="no_reference")

    ratio = price / reference
    if ratio > high_ratio:
        return PriceSanityResult(plausible=False, reason="above_band", ratio=ratio)
    if ratio < 1 / low_ratio:
        return PriceSanityResult(plausible=False, reason="below_band", ratio=ratio)
    return PriceSanityResult(plausible=True, reason="ok", ratio=ratio)
