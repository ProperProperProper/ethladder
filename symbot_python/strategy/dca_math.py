"""DCA order-sizing math.

Pure functions only — no exchange or DB access. Every price/qty a caller
passes in or receives back is expected to already be (or about to be)
rounded through the exchange's own instrument precision (see
exchange/bybit_client.py); this module does not know about instrument
metadata itself.

Two fee conventions are used on purpose:
- calculate_adjustments grosses up size by the ROUND-TRIP fee (2x the
  configured one-leg exchange_fee), so a filled rung already covers both
  the buy and the eventual sell commission.
- calculate_target_price adds only a SINGLE leg of fee to the take-profit
  percentage. These are independent, deliberate design choices and are
  not a bug to "fix" here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

PriceFilter = Callable[[float], float]
AmountFilter = Callable[[float], float]


def get_deviation_dca(step_percent: float, step_multiplier: float, n: int) -> float:
    """Cumulative % deviation from the base order price after n safety-order steps.

    n=0 -> 0. n=1 -> the first safety order's deviation. Etc.
    When step_multiplier == 1 the steps are flat and this is just n*step_percent;
    otherwise it's the sum of a geometric series of per-rung step percentages.
    """
    if n <= 0:
        return 0.0
    if step_multiplier == 1:
        return n * step_percent
    return step_percent * (1 - step_multiplier**n) / (1 - step_multiplier)


def filter_min_movement(value: float, min_move_amount: float) -> float:
    """Round `value` up to the nearest min_move_amount increment, then nudge
    it up by a small epsilon (5% of one increment) so float rounding never
    lands a quantity exactly on/under an exchange minimum.
    """
    if min_move_amount <= 0:
        return value
    steps = value / min_move_amount
    rounded = round(steps) * min_move_amount
    if rounded < value:
        rounded += min_move_amount
    return rounded + min_move_amount * 0.05


@dataclass
class AdjustedOrder:
    order_qty: float
    order_amount: float
    exchange_fee_qty: float
    exchange_fee_amount: float
    minimum_movement_amount: float


def calculate_adjustments(
    price: float,
    order_size: float,
    exchange_fee: float,
    min_move_amount: float,
    filter_amount: AmountFilter,
    filter_price: PriceFilter,
) -> AdjustedOrder:
    """Gross up a base-asset quantity to cover the round-trip (buy+sell) fee,
    then round through exchange precision.

    order_size is a base-asset quantity (e.g. how much BTC this rung buys
    before fee gross-up). Returns the fee-inclusive quantity/amount plus
    the fee components themselves, all exchange-precision filtered.
    """
    exchange_fee_roundtrip = exchange_fee * 2
    new_order_size = order_size
    fee_qty = 0.0

    # Escalate the fee estimate if it rounds to zero at exchange precision,
    # via a bounded retry-with-larger-fee loop.
    fee_multiplier = 1.0
    for _ in range(10):
        fee_qty = new_order_size / 100 * exchange_fee_roundtrip * fee_multiplier
        grossed = filter_min_movement(order_size + fee_qty, min_move_amount)
        grossed = filter_amount(grossed)
        if grossed > order_size or fee_qty == 0:
            new_order_size = grossed
            break
        fee_multiplier *= 1.25
    else:
        new_order_size = filter_amount(
            filter_min_movement(order_size + fee_qty, min_move_amount)
        )

    amount = filter_price(price * new_order_size)
    fee_amount = filter_price(price * fee_qty)

    return AdjustedOrder(
        order_qty=new_order_size,
        order_amount=amount,
        exchange_fee_qty=fee_qty,
        exchange_fee_amount=fee_amount,
        minimum_movement_amount=min_move_amount,
    )


def calculate_target_price(
    average: float,
    take_profit_percent: float,
    exchange_fee: float,
    filter_price: PriceFilter,
    price_tick: float,
    side: str = "long",
) -> float:
    """Take-profit target price.

    LONG: average lifted by (take_profit% + one leg of exchange fee),
    rounded UP to the nearest tick so the close can never undershoot the
    configured net profit.

    SHORT (mirror image): average lowered by the same amount, rounded
    DOWN to the nearest tick — same "never undershoot the configured net
    profit" guarantee, applied in the opposite direction.
    """
    if side == "short":
        exact_target = average * (1 - (take_profit_percent + exchange_fee) / 100)
        target_price = filter_price(exact_target)
        if target_price > exact_target and price_tick > 0:
            stepped = filter_price(target_price - price_tick)
            if stepped <= exact_target:
                target_price = stepped
        return target_price

    exact_target = average * (1 + (take_profit_percent + exchange_fee) / 100)
    target_price = filter_price(exact_target)
    if target_price < exact_target and price_tick > 0:
        stepped = filter_price(target_price + price_tick)
        if stepped >= exact_target:
            target_price = stepped
    return target_price


@dataclass
class ProfitResult:
    profit_percent: float
    profit_quote_projected: float
    current_profit_quote: float
    current_profit_base: float


def calculate_profit(
    price: float,
    order_average: float,
    order_sum: float,
    take_profit_percent: float,
    exchange_fee_percent: float,
    price_slippage_sell_percent: float,
    filter_amount: AmountFilter,
    side: str = "long",
    leverage: float = 1.0,
) -> ProfitResult:
    # LONG profits as price rises above average; SHORT profits as price
    # falls below average — mirror the raw percentage move accordingly.
    raw_move_percent = (
        (price - order_average) / order_average * 100
        if side == "long"
        else (order_average - price) / order_average * 100
    )
    raw_move_pct_after_costs = raw_move_percent - exchange_fee_percent - price_slippage_sell_percent
    # order_sum is cumulative NOTIONAL (qty * price summed across filled
    # rungs) — dividing by leverage recovers the margin actually
    # committed. profit_percent is reported as ROI on that margin, not
    # the raw price move, so it stays consistent with current_profit_quote
    # (which IS leverage-amplified) — otherwise a leveraged trade shows a
    # large dollar profit next to a misleadingly small "%", exactly the
    # mismatch a leveraged bot's own report must not produce.
    effective_leverage = leverage if leverage > 0 else 1.0
    margin = order_sum / effective_leverage
    current_profit_quote = round(order_sum * (raw_move_pct_after_costs / 100), 8)
    profit_percent = round((current_profit_quote / margin * 100) if margin else 0.0, 2)
    profit_quote_projected = round(order_sum * (take_profit_percent / 100), 8)
    current_profit_base = filter_amount(
        current_profit_quote / price if price else 0.0
    )
    return ProfitResult(
        profit_percent=profit_percent,
        profit_quote_projected=profit_quote_projected,
        current_profit_quote=current_profit_quote,
        current_profit_base=current_profit_base,
    )


def calculate_liquidation_price(
    average: float,
    leverage: float,
    maintenance_margin_rate: float = 0.005,
    side: str = "long",
) -> float:
    """Approximate isolated-margin liquidation price, ignoring funding
    payments: price moves against the position until the margin posted
    (1/leverage of notional) is consumed down to the maintenance margin
    requirement.

    LONG:  liq_price = average * (1 - 1/leverage + maintenance_margin_rate)
           (price falls to trigger it)
    SHORT: liq_price = average * (1 + 1/leverage - maintenance_margin_rate)
           (price rises to trigger it — mirror image)

    leverage <= 1 has no liquidation risk (fully margined / spot-equivalent).
    """
    if leverage <= 1:
        return 0.0
    if side == "short":
        # A short's liq price must never fall below its own average — if
        # maintenance_margin_rate exceeds 1/leverage (a degenerate/
        # already-under-margin combination), the raw formula computes a
        # price BELOW average, which would nonsensically imply a short
        # gets liquidated by a FAVORABLE (downward) move. Clamped to
        # `average`, mirroring the long side's `max(liq, 0.0)` guard
        # against the same class of degenerate leverage/mmr combination.
        return max(average * (1 + 1 / leverage - maintenance_margin_rate), average)
    liq = average * (1 - 1 / leverage + maintenance_margin_rate)
    return max(liq, 0.0)


def calculate_max_funds(
    first_order_amount: float,
    dca_order_amount: float,
    dca_max_order: int,
    dca_order_size_multiplier: float,
    exchange_fee: float,
) -> float:
    """Theoretical total quote-currency capital required to fill every rung.
    Display/sizing-preview only — never used as a live trading gate. Uses
    only the buy-side fee once (not round-tripped) since this is an
    upfront capital estimate, not a fee-inclusive fill calculation.
    """
    fee_factor = 1 + exchange_fee / 100
    total = first_order_amount * fee_factor
    for i in range(dca_max_order):
        total += dca_order_amount * (dca_order_size_multiplier**i) * fee_factor
    return total


def solve_order_sizing_for_budget(
    first_order_amount: float,
    dca_order_amount: float,
    dca_max_order: int,
    dca_order_size_multiplier: float,
    exchange_fee: float,
    target_budget: float,
) -> tuple[float, float]:
    """Rescale (first_order_amount, dca_order_amount) proportionally so
    calculate_max_funds(...) exactly equals target_budget, preserving
    whatever ratio the caller configured between the base order and the
    safety-order sizing.

    This is how "always deploy X% of available funds" is implemented:
    calculate_max_funds is linear in both amounts when they're scaled
    together by the same factor, so there's a single closed-form scale
    that hits the target exactly — no search needed.

    If the unscaled ladder requires zero funds (a degenerate config),
    the inputs are returned unchanged rather than dividing by zero.
    """
    unit_max_funds = calculate_max_funds(
        first_order_amount, dca_order_amount, dca_max_order, dca_order_size_multiplier, exchange_fee
    )
    if unit_max_funds <= 0:
        return first_order_amount, dca_order_amount
    scale = target_budget / unit_max_funds
    return first_order_amount * scale, dca_order_amount * scale


@dataclass
class OrderRung:
    price: float
    qty: float
    amount: float
    qty_sum: float
    sum: float
    average: float
    target: float
    filled: int = 0
    manual: bool = False


def recalculate_orders(
    orders: list[OrderRung],
    changed_index: Optional[int],
    exchange_fee: float,
    min_move_amount: float,
    take_profit_percent: float,
    filter_amount: AmountFilter,
    filter_price: PriceFilter,
    price_tick: float,
    side: str = "long",
) -> list[OrderRung]:
    """Recompute cumulative qty_sum/sum/average/target for every rung from
    `changed_index` onward (or from the start if None), carrying forward
    the running totals. A rung that is both filled and manual (a real
    executed fill, e.g. from a partial-fill credit) is frozen: its own
    qty/amount/qty_sum/sum are never recomputed, only used as the running
    total's starting point for rungs after it.

    average = cumulative_quote_spent / cumulative_base_qty_held (VWAP).
    """
    # Always walk from the top to rebuild the running totals correctly;
    # changed_index only tells the caller which single rung's raw
    # price/qty/amount to re-derive via calculate_adjustments before this
    # pass (that re-derivation is the caller's responsibility).
    del changed_index
    running_qty_sum = 0.0
    running_sum = 0.0
    for i, rung in enumerate(orders):
        if rung.filled and rung.manual:
            running_qty_sum = rung.qty_sum
            running_sum = rung.sum
            continue
        qty_sum = filter_amount(running_qty_sum + rung.qty)
        total = round(filter_price(running_sum + rung.amount), 8)
        average = filter_price(total / qty_sum) if qty_sum else 0.0
        target = calculate_target_price(
            average, take_profit_percent, exchange_fee, filter_price, price_tick, side
        )
        orders[i] = OrderRung(
            price=rung.price,
            qty=rung.qty,
            amount=rung.amount,
            qty_sum=qty_sum,
            sum=total,
            average=average,
            target=target,
            filled=rung.filled,
            manual=rung.manual,
        )
        running_qty_sum, running_sum = qty_sum, total
    return orders
