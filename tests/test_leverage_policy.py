import pytest

from symbot_python.strategy.leverage_policy import (
    DEFAULT_LEVERAGE,
    MAX_LEVERAGE,
    MIN_LEVERAGE,
    clamp_leverage,
)


def test_min_max_default_are_the_locked_band():
    assert MIN_LEVERAGE == 9.0
    assert MAX_LEVERAGE == 11.0
    assert MIN_LEVERAGE <= DEFAULT_LEVERAGE <= MAX_LEVERAGE


def test_clamp_leverage_within_band_is_unchanged():
    assert clamp_leverage(10.0) == pytest.approx(10.0)


def test_clamp_leverage_above_max_is_capped():
    assert clamp_leverage(20.0) == pytest.approx(MAX_LEVERAGE)
    assert clamp_leverage(11.0001) == pytest.approx(MAX_LEVERAGE)


def test_clamp_leverage_below_min_is_floored():
    assert clamp_leverage(1.0) == pytest.approx(MIN_LEVERAGE)
    assert clamp_leverage(8.9999) == pytest.approx(MIN_LEVERAGE)


def test_clamp_leverage_at_exact_boundaries_is_unchanged():
    assert clamp_leverage(MIN_LEVERAGE) == pytest.approx(MIN_LEVERAGE)
    assert clamp_leverage(MAX_LEVERAGE) == pytest.approx(MAX_LEVERAGE)
