import pytest

from symbot_python.strategy import dca_math as m


def identity(x: float) -> float:
    return x


def round8(x: float) -> float:
    return round(x, 8)


def test_get_deviation_dca_flat_steps():
    # step_multiplier == 1 -> plain linear cumulative deviation
    assert m.get_deviation_dca(1.3, 1.0, 0) == 0
    assert m.get_deviation_dca(1.3, 1.0, 1) == pytest.approx(1.3)
    assert m.get_deviation_dca(1.3, 1.0, 5) == pytest.approx(6.5)


def test_get_deviation_dca_geometric_steps():
    # step_multiplier != 1 -> geometric series sum
    step, mult, n = 1.0, 1.1, 3
    expected = step * (1 - mult**n) / (1 - mult)
    assert m.get_deviation_dca(step, mult, n) == pytest.approx(expected)


def test_filter_min_movement_rounds_up_and_nudges():
    # value already on a boundary should still nudge up slightly
    result = m.filter_min_movement(1.0, 0.01)
    assert result > 1.0


def test_calculate_target_price_rounds_up_to_tick():
    # average=100, take_profit=1.5%, fee=0.45% -> exact target = 100*(1+0.0195)=101.95
    # filter_price simulates truncation to 2 tick units below exact (e.g. floors to 101.9)
    def floor_to_tick(p: float) -> float:
        tick = 0.1
        return (int(p / tick)) * tick

    target = m.calculate_target_price(
        average=100.0,
        take_profit_percent=1.5,
        exchange_fee=0.45,
        filter_price=floor_to_tick,
        price_tick=0.1,
    )
    exact = 100.0 * (1 + (1.5 + 0.45) / 100)
    assert target >= exact - 1e-9


def test_calculate_target_price_exact_when_filter_is_identity():
    target = m.calculate_target_price(
        average=100.0,
        take_profit_percent=1.5,
        exchange_fee=0.45,
        filter_price=identity,
        price_tick=0.01,
    )
    assert target == pytest.approx(101.95)


def test_calculate_adjustments_grosses_up_for_roundtrip_fee():
    result = m.calculate_adjustments(
        price=100.0,
        order_size=1.0,
        exchange_fee=0.45,
        min_move_amount=0.0001,
        filter_amount=identity,
        filter_price=round8,
    )
    # 1.0 base qty grossed up by ~0.9% (2x0.45) roundtrip fee, plus a tiny nudge
    assert result.order_qty > 1.0
    assert result.order_qty < 1.02
    assert result.exchange_fee_qty > 0


def test_calculate_profit_basic():
    result = m.calculate_profit(
        price=110.0,
        order_average=100.0,
        order_sum=1000.0,
        take_profit_percent=1.5,
        exchange_fee_percent=0.45,
        price_slippage_sell_percent=0.0,
        filter_amount=identity,
    )
    # (110-100)/100*100 - 0.45 = 9.55
    assert result.profit_percent == pytest.approx(9.55)
    assert result.current_profit_quote == pytest.approx(95.5)


def test_calculate_max_funds_single_leg_fee_only():
    total = m.calculate_max_funds(
        first_order_amount=20,
        dca_order_amount=45,
        dca_max_order=2,
        dca_order_size_multiplier=1.08,
        exchange_fee=0.45,
    )
    fee_factor = 1.0045
    expected = 20 * fee_factor + 45 * fee_factor + 45 * 1.08 * fee_factor
    assert total == pytest.approx(expected)


def test_recalculate_orders_vwap():
    orders = [
        m.OrderRung(price=100, qty=1.0, amount=100.0, qty_sum=0, sum=0, average=0, target=0),
        m.OrderRung(price=90, qty=2.0, amount=180.0, qty_sum=0, sum=0, average=0, target=0),
    ]
    result = m.recalculate_orders(
        orders,
        changed_index=None,
        exchange_fee=0.0,
        min_move_amount=0.0001,
        take_profit_percent=1.5,
        filter_amount=identity,
        filter_price=identity,
        price_tick=0.0,
    )
    # rung 0: qty_sum=1, sum=100, average=100
    assert result[0].qty_sum == pytest.approx(1.0)
    assert result[0].average == pytest.approx(100.0)
    # rung 1: qty_sum=3, sum=280, average=280/3
    assert result[1].qty_sum == pytest.approx(3.0)
    assert result[1].sum == pytest.approx(280.0)
    assert result[1].average == pytest.approx(280.0 / 3.0)


def test_recalculate_orders_freezes_filled_manual_rung():
    orders = [
        m.OrderRung(price=100, qty=1.0, amount=100.0, qty_sum=1.0, sum=100.0, average=100.0, target=101.5, filled=1, manual=True),
        m.OrderRung(price=90, qty=2.0, amount=180.0, qty_sum=0, sum=0, average=0, target=0),
    ]
    result = m.recalculate_orders(
        orders,
        changed_index=None,
        exchange_fee=0.0,
        min_move_amount=0.0001,
        take_profit_percent=1.5,
        filter_amount=identity,
        filter_price=identity,
        price_tick=0.0,
    )
    # frozen rung unchanged
    assert result[0].qty_sum == pytest.approx(1.0)
    assert result[0].sum == pytest.approx(100.0)
    # second rung's running total starts from the frozen rung's values
    assert result[1].qty_sum == pytest.approx(3.0)
    assert result[1].sum == pytest.approx(280.0)


def test_calculate_liquidation_price_spot_leverage_one_has_no_liquidation():
    assert m.calculate_liquidation_price(100.0, leverage=1.0) == 0.0
    assert m.calculate_liquidation_price(100.0, leverage=0.5) == 0.0  # below 1x is also "no liquidation"


def test_calculate_liquidation_price_11x():
    # liq = average * (1 - 1/11 + mmr)
    liq = m.calculate_liquidation_price(100.0, leverage=11.0, maintenance_margin_rate=0.005)
    expected = 100.0 * (1 - 1 / 11 + 0.005)
    assert liq == pytest.approx(expected)
    assert liq == pytest.approx(91.41, abs=0.01)


def test_calculate_liquidation_price_scales_with_average():
    liq_100 = m.calculate_liquidation_price(100.0, leverage=10.0)
    liq_200 = m.calculate_liquidation_price(200.0, leverage=10.0)
    assert liq_200 == pytest.approx(liq_100 * 2)


def test_calculate_liquidation_price_never_negative():
    # absurdly high leverage shouldn't produce a negative price
    liq = m.calculate_liquidation_price(100.0, leverage=1000.0, maintenance_margin_rate=0.0)
    assert liq >= 0.0


def test_solve_order_sizing_for_budget_hits_target_exactly():
    first, dca = m.solve_order_sizing_for_budget(
        first_order_amount=20.0, dca_order_amount=45.0, dca_max_order=10,
        dca_order_size_multiplier=1.08, exchange_fee=0.1, target_budget=980.0,
    )
    achieved = m.calculate_max_funds(first, dca, 10, 1.08, 0.1)
    assert achieved == pytest.approx(980.0)


def test_solve_order_sizing_for_budget_preserves_ratio():
    original_ratio = 45.0 / 20.0
    first, dca = m.solve_order_sizing_for_budget(
        first_order_amount=20.0, dca_order_amount=45.0, dca_max_order=10,
        dca_order_size_multiplier=1.08, exchange_fee=0.1, target_budget=5000.0,
    )
    assert dca / first == pytest.approx(original_ratio)


def test_solve_order_sizing_for_budget_scales_up_and_down():
    small_first, small_dca = m.solve_order_sizing_for_budget(20.0, 45.0, 10, 1.08, 0.1, target_budget=100.0)
    big_first, big_dca = m.solve_order_sizing_for_budget(20.0, 45.0, 10, 1.08, 0.1, target_budget=100_000.0)
    assert small_first < 20.0
    assert big_first > 20.0
    assert big_dca > small_dca


def test_solve_order_sizing_for_budget_zero_unit_funds_returns_inputs_unchanged():
    first, dca = m.solve_order_sizing_for_budget(0.0, 0.0, 5, 1.0, 0.0, target_budget=1000.0)
    assert (first, dca) == (0.0, 0.0)


# -- short-side mirror tests --------------------------------------------------


def test_calculate_target_price_short_is_below_average():
    target = m.calculate_target_price(
        average=100.0, take_profit_percent=1.5, exchange_fee=0.45,
        filter_price=identity, price_tick=0.01, side="short",
    )
    # short target = average * (1 - (1.5+0.45)/100) = 98.05
    assert target == pytest.approx(98.05)


def test_calculate_target_price_short_rounds_down_never_overshoots():
    def floor_to_tick(p: float) -> float:
        tick = 0.1
        return (int(p / tick)) * tick

    target = m.calculate_target_price(
        average=100.0, take_profit_percent=1.5, exchange_fee=0.45,
        filter_price=floor_to_tick, price_tick=0.1, side="short",
    )
    exact = 100.0 * (1 - (1.5 + 0.45) / 100)
    # target must be <= exact (never require MORE favorable a price than configured)
    assert target <= exact + 1e-9


def test_calculate_profit_short_profits_when_price_falls():
    result = m.calculate_profit(
        price=90.0, order_average=100.0, order_sum=1000.0,
        take_profit_percent=1.5, exchange_fee_percent=0.45,
        price_slippage_sell_percent=0.0, filter_amount=identity, side="short",
    )
    # (100-90)/100*100 - 0.45 = 9.55
    assert result.profit_percent == pytest.approx(9.55)
    assert result.current_profit_quote == pytest.approx(95.5)


def test_calculate_profit_short_loses_when_price_rises():
    result = m.calculate_profit(
        price=110.0, order_average=100.0, order_sum=1000.0,
        take_profit_percent=1.5, exchange_fee_percent=0.0,
        price_slippage_sell_percent=0.0, filter_amount=identity, side="short",
    )
    assert result.profit_percent < 0


def test_calculate_liquidation_price_short_is_above_average():
    liq = m.calculate_liquidation_price(100.0, leverage=11.0, maintenance_margin_rate=0.005, side="short")
    expected = 100.0 * (1 + 1 / 11 - 0.005)
    assert liq == pytest.approx(expected)
    assert liq > 100.0


def test_calculate_liquidation_price_short_leverage_one_has_no_liquidation():
    assert m.calculate_liquidation_price(100.0, leverage=1.0, side="short") == 0.0


def test_calculate_liquidation_price_short_never_below_average():
    # Degenerate case: maintenance_margin_rate (0.6) exceeds 1/leverage
    # (0.5 at leverage=2) — the raw textbook formula
    # average*(1 + 1/leverage - mmr) computes to average*0.9, BELOW
    # average, which would nonsensically mean a short gets liquidated by
    # a FAVORABLE (downward) price move. Must clamp to average instead.
    liq = m.calculate_liquidation_price(100.0, leverage=2.0, maintenance_margin_rate=0.6, side="short")
    assert liq >= 100.0


def test_recalculate_orders_short_targets_below_average():
    orders = [
        m.OrderRung(price=100, qty=1.0, amount=100.0, qty_sum=0, sum=0, average=0, target=0),
    ]
    result = m.recalculate_orders(
        orders, changed_index=None, exchange_fee=0.0, min_move_amount=0.0001,
        take_profit_percent=2.0, filter_amount=identity, filter_price=identity,
        price_tick=0.0, side="short",
    )
    assert result[0].average == pytest.approx(100.0)
    assert result[0].target == pytest.approx(98.0)
    assert result[0].target < result[0].average
