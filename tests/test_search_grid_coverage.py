"""Regression coverage for the SEARCH_GRID/WINNER_PARAM_FIELDS gaps found
in review: a param that's searched but never reaches live/paper is dead
weight (the optimizer can pick it, but the winner sync silently drops it
on the floor), and a bool-enabled param whose companion value is never
searched degenerates to the BacktestConfig field default (e.g.
dca_stop_loss_percent defaulting to 0.0 -> stops out on any adverse tick).
"""

import run_everything as co
from symbot_python.api.paper import WINNER_PARAM_FIELDS
from symbot_python.strategy.backtest import BacktestConfig, FIXED_TAKE_PROFIT_PERCENT
from symbot_python.strategy.models import BotConfig


def test_every_searched_field_is_a_real_backtest_config_field():
    valid_fields = {f.name for f in BacktestConfig.__dataclass_fields__.values()}
    for field in co.SEARCH_GRID.values:
        assert field in valid_fields, f"{field} is searched but not a BacktestConfig field"


def test_take_profit_is_fixed_and_not_an_optimizer_dimension():
    assert "dca_take_profit_percent" not in co.SEARCH_GRID.values
    assert co.make_base_config(0.1, 0.001, [], []).dca_take_profit_percent == FIXED_TAKE_PROFIT_PERCENT


def test_every_searched_field_reaches_live_paper_via_winner_param_fields():
    for field in co.SEARCH_GRID.values:
        assert field in WINNER_PARAM_FIELDS, (
            f"{field} is searched but missing from WINNER_PARAM_FIELDS — a promoted "
            "winner using it would never actually reach the live/paper bot"
        )


def test_every_winner_param_field_is_a_real_bot_config_field():
    valid_fields = {f.name for f in BotConfig.__dataclass_fields__.values()}
    for field in WINNER_PARAM_FIELDS:
        assert field in valid_fields, f"{field} is in WINNER_PARAM_FIELDS but not a BotConfig field"


def test_stop_loss_percent_and_trailing_and_reversal_fields_are_searched():
    # These three were the concrete gaps: dca_stop_loss_percent silently
    # defaulted to 0.0 whenever dca_stop_loss_enabled=True was sampled,
    # trailing-stop was never searched at all, and reverse_drawdown_percent
    # only existed on the live engine (not the backtest/walk-forward side).
    for field in (
        "dca_stop_loss_percent",
        "dca_trailing_stop_enabled",
        "dca_trailing_stop_distance",
        "dca_trailing_activate_profit",
        "reverse_drawdown_percent",
        "reverse_cooldown_sec",
        "max_consecutive_reversals",
    ):
        assert field in co.SEARCH_GRID.values
        assert len(co.SEARCH_GRID.values[field]) > 1


def test_reverse_drawdown_percent_grid_includes_disabled_as_an_option():
    # None must remain a sampleable value so the search can discover that
    # NOT reversing is best for a given interval/window, rather than
    # forcing every sampled candidate to use stop-and-reverse.
    assert None in co.SEARCH_GRID.values["reverse_drawdown_percent"]


def test_a_sampled_combination_builds_a_valid_backtest_config():
    combo = co.SEARCH_GRID.sample(1)[0]
    config = BacktestConfig(
        first_order_amount=20.0, dca_order_amount=45.0, price_tick=0.1, min_move_amount=0.001,
        dca_order_start_distance=0.5, dca_take_profit_percent=FIXED_TAKE_PROFIT_PERCENT, exchange_fee=0.06,
        **combo,
    )
    for field, value in combo.items():
        assert getattr(config, field) == value
