import pytest

from symbot_python.strategy.stop_loss import StopLossInput, evaluate


def base_input(**overrides) -> StopLossInput:
    defaults = dict(
        enabled=True,
        price=100.0,
        average=100.0,
        stop_loss_percent=5.0,
        reference="average",
        last_safety_order_price=0.0,
        fee_rate=0.45,
        move_breakeven=False,
        breakeven_trigger=None,
        profit_percentage=0.0,
        breakeven_armed=False,
        active_stop_loss_price=0.0,
        trailing_enabled=False,
        trailing_distance=0.0,
        trailing_activate_profit=0.0,
        trail_high_price=0.0,
    )
    defaults.update(overrides)
    return StopLossInput(**defaults)


def test_disabled_never_triggers():
    result = evaluate(base_input(enabled=False, stop_loss_percent=0, trailing_enabled=False))
    assert result.triggered is False
    assert result.reason == "disabled"


def test_base_stop_not_hit_above_level():
    # average=100, sl=5% -> level=95; price=96 should not trigger
    result = evaluate(base_input(price=96.0))
    assert result.triggered is False
    assert result.level == 95.0


def test_base_stop_hit_at_or_below_level():
    result = evaluate(base_input(price=95.0))
    assert result.triggered is True
    assert result.hit_label == "base"
    assert result.level == 95.0


def test_last_safety_order_reference():
    result = evaluate(
        base_input(reference="lastSafetyOrder", last_safety_order_price=80.0, price=76.0)
    )
    # level = 80 * 0.95 = 76
    assert result.triggered is True
    assert result.level == 76.0


def test_breakeven_arms_when_profit_hits_trigger():
    result = evaluate(
        base_input(
            move_breakeven=True,
            breakeven_trigger=1.0,
            profit_percentage=1.5,
            price=101.0,  # above the newly-armed breakeven level (100.9)
        )
    )
    assert result.breakeven_armed is True
    assert result.triggered is False
    assert result.reason == "armed_breakeven"
    # breakeven_level = 100 * (1 + 2*0.45/100) = 100.9
    assert result.breakeven_level == pytest.approx(100.9)


def test_breakeven_triggers_when_armed_and_price_falls_to_level():
    result = evaluate(
        base_input(
            move_breakeven=True,
            breakeven_armed=True,
            breakeven_trigger=1.0,
            profit_percentage=1.5,
            price=100.5,  # clearly below the ~100.9 breakeven level
        )
    )
    assert result.triggered is True
    assert result.hit_label == "breakeven"


def test_trailing_stop_triggers_and_takes_precedence_over_base():
    result = evaluate(
        base_input(
            stop_loss_percent=5.0,
            trailing_enabled=True,
            trailing_distance=2.0,
            trailing_activate_profit=1.0,
            profit_percentage=5.0,
            trail_high_price=110.0,
            price=107.7,  # trail_level = 110*0.98 = 107.8 -> price below triggers
        )
    )
    assert result.triggered is True
    assert result.hit_label == "trailing"


def test_ratchet_value_persists_even_if_base_would_be_lower():
    # average drifted down so base_stop_level (95*.., recompute) would be
    # lower than a previously-persisted higher active_stop_loss_price;
    # the higher ratchet value must still win (never moves down).
    result = evaluate(
        base_input(
            average=90.0,  # base level would be 90*0.95=85.5
            active_stop_loss_price=97.0,
            price=98.0,
        )
    )
    assert result.triggered is False
    assert result.level == 97.0


def test_no_reference_when_stoploss_usable_but_average_missing():
    result = evaluate(base_input(average=0.0, price=50.0))
    assert result.triggered is False
    assert result.reason == "no_reference"


def test_invalid_price_fails_safe():
    result = evaluate(base_input(price=0.0))
    assert result.triggered is False
    assert result.reason == "no_reference"


# -- short-side mirror tests --------------------------------------------------


def test_short_base_stop_not_hit_below_level():
    # average=100, sl=5% -> level=105 (ceiling); price=104 should not trigger
    result = evaluate(base_input(side="short", price=104.0))
    assert result.triggered is False
    assert result.level == 105.0


def test_short_base_stop_hit_at_or_above_level():
    result = evaluate(base_input(side="short", price=105.0))
    assert result.triggered is True
    assert result.hit_label == "base"
    assert result.level == 105.0


def test_short_last_safety_order_reference():
    result = evaluate(
        base_input(side="short", reference="lastSafetyOrder", last_safety_order_price=80.0, price=84.0)
    )
    # level = 80 * 1.05 = 84
    assert result.triggered is True
    assert result.level == 84.0


def test_short_breakeven_arms_when_profit_hits_trigger():
    result = evaluate(
        base_input(
            side="short", move_breakeven=True, breakeven_trigger=1.0, profit_percentage=1.5,
            price=98.0,  # below the newly-armed breakeven level (~99.1)
        )
    )
    assert result.breakeven_armed is True
    assert result.triggered is False
    assert result.reason == "armed_breakeven"
    # breakeven_level = 100 * (1 - 2*0.45/100) = 99.1
    assert result.breakeven_level == pytest.approx(99.1)


def test_short_breakeven_triggers_when_armed_and_price_rises_to_level():
    result = evaluate(
        base_input(
            side="short", move_breakeven=True, breakeven_armed=True, breakeven_trigger=1.0,
            profit_percentage=1.5, price=99.5,  # above the ~99.1 breakeven ceiling
        )
    )
    assert result.triggered is True
    assert result.hit_label == "breakeven"


def test_short_trailing_stop_triggers_and_takes_precedence_over_base():
    result = evaluate(
        base_input(
            side="short", stop_loss_percent=5.0, trailing_enabled=True, trailing_distance=2.0,
            trailing_activate_profit=1.0, profit_percentage=5.0,
            trail_high_price=90.0,  # this is the trailing LOW for a short
            price=91.9,  # trail_level = 90*1.02 = 91.8 -> price above triggers
        )
    )
    assert result.triggered is True
    assert result.hit_label == "trailing"


def test_short_ratchet_value_persists_even_if_base_would_be_higher():
    # average drifted up so base_stop_level (110*1.05=115.5) would be
    # higher (looser) than a previously-persisted lower ceiling; the
    # lower ratchet value must still win (a short's ceiling never loosens).
    result = evaluate(
        base_input(side="short", average=110.0, active_stop_loss_price=103.0, price=102.0)
    )
    assert result.triggered is False
    assert result.level == 103.0


def test_short_invalid_price_fails_safe():
    result = evaluate(base_input(side="short", price=0.0))
    assert result.triggered is False
    assert result.reason == "no_reference"
