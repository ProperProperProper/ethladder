import pytest

from symbot_python.signals.candles import DEFAULT_BACKTEST_DAYS, bars_for_days, interval_minutes


def test_default_backtest_days_is_locked_to_14():
    assert DEFAULT_BACKTEST_DAYS == 14


def test_interval_minutes_native():
    assert interval_minutes("1") == 1
    assert interval_minutes("60") == 60
    assert interval_minutes("D") == 1440
    assert interval_minutes("W") == 10080


def test_interval_minutes_custom():
    assert interval_minutes("29") == 29
    assert interval_minutes("59") == 59


def test_bars_for_days_hourly_14_days():
    assert bars_for_days("60") == 336  # 14*24


def test_bars_for_days_daily_14_days():
    assert bars_for_days("D") == 14


def test_bars_for_days_custom_29m_14_days():
    assert bars_for_days("29") == pytest.approx(14 * 24 * 60 / 29, abs=1)


def test_bars_for_days_scales_with_requested_days():
    assert bars_for_days("60", days=7) == 168
    assert bars_for_days("60", days=28) == 672


def test_bars_for_days_rounds_up_never_short():
    # 1000 minutes doesn't divide evenly into 14 days (20160 min) -> must
    # round UP so the caller never gets slightly less than 14 days.
    bars = bars_for_days("1000")
    assert bars * 1000 >= 14 * 24 * 60
