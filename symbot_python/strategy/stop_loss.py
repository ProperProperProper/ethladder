"""Stop-loss / break-even / trailing-stop decision engine, implemented
for the long side and mirrored here for short.

Pure function, no I/O: evaluate() takes a snapshot of a deal's current
price/average/persisted ratchet state and returns whether to close the
deal now, and what the (possibly unchanged) ratchet state should become.
The caller is responsible for actually persisting activeStopLossPrice /
trailHighPrice / breakevenArmed changes — including, for a SHORT
position, only ever RATCHETING active_stop_loss_price DOWN (mirroring
the long side's ratchet-up-only), since 0.0 remains the universal
"not yet set" sentinel for both directions (real prices are always > 0).

trail_high_price is the trailing reference extreme regardless of side:
for a long it's the highest price seen since trailing activated; for a
short it's the LOWEST price seen. The caller is responsible for tracking
it in the correct direction (max() for long, min() for short) — this
module only ever reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Reference = Literal["average", "lastSafetyOrder"]
DEFAULT_REFERENCE: Reference = "average"

Side = Literal["long", "short"]

Reason = Literal[
    "disabled",
    "no_reference",
    "inactive",
    "stop_hit",
    "armed_breakeven",
    "trailing_active",
    "ok",
]
HitLabel = Literal["base", "breakeven", "trailing"]


@dataclass
class StopLossInput:
    enabled: bool
    price: float
    average: float
    stop_loss_percent: float = 0.0
    reference: Reference = DEFAULT_REFERENCE
    last_safety_order_price: float = 0.0
    fee_rate: float = 0.0
    move_breakeven: bool = False
    breakeven_trigger: Optional[float] = None
    profit_percentage: float = 0.0
    breakeven_armed: bool = False
    active_stop_loss_price: float = 0.0
    trailing_enabled: bool = False
    trailing_distance: float = 0.0
    trailing_activate_profit: float = 0.0
    trail_high_price: float = 0.0
    side: Side = "long"


@dataclass
class StopLossResult:
    triggered: bool
    level: float = 0.0
    breakeven_armed: bool = False
    breakeven_level: float = 0.0
    trailing_active: bool = False
    trail_level: float = 0.0
    reason: Reason = "inactive"
    hit_label: Optional[HitLabel] = None
    message: str = ""


def evaluate(data: StopLossInput) -> StopLossResult:
    is_short = data.side == "short"
    sl_usable = data.enabled and data.stop_loss_percent > 0
    trailing_configured = data.trailing_enabled and data.trailing_distance > 0

    if not sl_usable and not trailing_configured:
        return StopLossResult(triggered=False, reason="disabled", message="Stop-loss and trailing both disabled")

    if data.price <= 0:
        return StopLossResult(triggered=False, reason="no_reference", message="No usable current price")

    base_stop_level: Optional[float] = None
    if sl_usable:
        ref_price = (
            data.last_safety_order_price
            if data.reference == "lastSafetyOrder"
            else data.average
        )
        if ref_price and ref_price > 0:
            base_stop_level = (
                ref_price * (1 + data.stop_loss_percent / 100)
                if is_short
                else ref_price * (1 - data.stop_loss_percent / 100)
            )
        elif not trailing_configured:
            return StopLossResult(
                triggered=False, reason="no_reference", message="No reference price for stop-loss"
            )

    if data.fee_rate > 0:
        breakeven_level = (
            data.average * (1 - 2 * (data.fee_rate / 100))
            if is_short
            else data.average * (1 + 2 * (data.fee_rate / 100))
        )
    else:
        breakeven_level = data.average

    breakeven_armed = data.breakeven_armed
    newly_armed = False
    if (
        data.enabled
        and data.move_breakeven
        and not breakeven_armed
        and breakeven_level
        and data.breakeven_trigger is not None
        and data.profit_percentage >= data.breakeven_trigger
    ):
        breakeven_armed = True
        newly_armed = True

    trailing_active = (
        trailing_configured
        and data.profit_percentage >= data.trailing_activate_profit
        and data.trail_high_price > 0
    )
    trail_level = (
        (
            data.trail_high_price * (1 + data.trailing_distance / 100)
            if is_short
            else data.trail_high_price * (1 - data.trailing_distance / 100)
        )
        if trailing_active
        else 0.0
    )

    candidates: list[tuple[HitLabel, float]] = []
    if base_stop_level is not None:
        candidates.append(("base", base_stop_level))
    if breakeven_armed:
        candidates.append(("breakeven", breakeven_level))
    if trailing_active:
        candidates.append(("trailing", trail_level))
    if data.active_stop_loss_price > 0:
        # The persisted ratchet value participates unconditionally so the
        # effective level never loosens tick-to-tick even if e.g. the
        # average has since drifted and base_stop_level would be less
        # protective now. Long ratchets up-only; short ratchets down-only
        # (enforced by the caller when deciding whether to persist a new
        # value, not here).
        candidates.append(("base", data.active_stop_loss_price))

    if not candidates:
        return StopLossResult(
            triggered=False,
            breakeven_armed=breakeven_armed,
            breakeven_level=breakeven_level,
            trailing_active=trailing_active,
            trail_level=trail_level,
            reason="inactive",
            message="No active stop level",
        )

    # Precedence on ties: trailing > breakeven > base.
    precedence = {"trailing": 2, "breakeven": 1, "base": 0}
    if is_short:
        # The protective ceiling ratchets DOWN as it tightens, so the
        # tightest (lowest) candidate is the effective level.
        best_label, level = min(candidates, key=lambda c: (c[1], -precedence[c[0]]))
        triggered = data.price >= level
    else:
        best_label, level = max(candidates, key=lambda c: (c[1], precedence[c[0]]))
        triggered = data.price <= level

    if triggered:
        return StopLossResult(
            triggered=True,
            level=level,
            breakeven_armed=breakeven_armed,
            breakeven_level=breakeven_level,
            trailing_active=trailing_active,
            trail_level=trail_level,
            reason="stop_hit",
            hit_label=best_label,
            message=f"Stop hit ({best_label}) at {level}",
        )

    if newly_armed:
        reason: Reason = "armed_breakeven"
    elif trailing_active:
        reason = "trailing_active"
    else:
        reason = "ok"

    return StopLossResult(
        triggered=False,
        level=level,
        breakeven_armed=breakeven_armed,
        breakeven_level=breakeven_level,
        trailing_active=trailing_active,
        trail_level=trail_level,
        reason=reason,
        message="Stop active but not hit",
    )


def to_bool(value: object) -> bool:
    return value in (True, "true", 1, "1")
