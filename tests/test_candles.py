import pytest

from symbot_python.signals.candles import is_native_interval, resample_candles


def make_1m_candles(count: int, start_ts_ms: int = 0):
    """count consecutive 1-minute candles, price rising by 1 each minute."""
    return [
        [start_ts_ms + i * 60_000, 100.0 + i, 100.5 + i, 99.5 + i, 100.2 + i, 10.0]
        for i in range(count)
    ]


def test_is_native_interval():
    assert is_native_interval("60") is True
    assert is_native_interval("D") is True
    assert is_native_interval("29") is False


def test_resample_empty_input():
    assert resample_candles([], 29) == []


def test_resample_bucket_minutes_zero_returns_input_unchanged():
    candles = make_1m_candles(5)
    assert resample_candles(candles, 0) == candles


def test_resample_29m_bucket_aggregates_correctly():
    candles = make_1m_candles(58)  # exactly two 29-minute buckets, epoch-aligned
    result = resample_candles(candles, 29)
    assert len(result) == 2

    first, second = result
    # first bucket: minutes 0..28
    assert first[0] == 0
    assert first[1] == candles[0][1]  # open = first candle's open
    assert first[2] == max(c[2] for c in candles[0:29])  # high
    assert first[3] == min(c[3] for c in candles[0:29])  # low
    assert first[4] == candles[28][4]  # close = last candle's close
    assert first[5] == pytest.approx(sum(c[5] for c in candles[0:29]))

    # second bucket: minutes 29..57
    assert second[0] == 29 * 60_000
    assert second[4] == candles[57][4]


def test_resample_alignment_is_epoch_based_not_input_based():
    # Start mid-way through a would-be bucket to prove alignment is fixed
    # to epoch boundaries, not to wherever the input happens to start.
    start = 10 * 60_000  # 10 minutes past epoch
    candles = make_1m_candles(20, start_ts_ms=start)
    result = resample_candles(candles, 29)
    # All 20 candles (minutes 10..29) fall within the single epoch-aligned
    # bucket [0, 29) union... minute 29 itself starts a NEW bucket.
    assert result[0][0] == 0
    # minute 29 (the 20th candle, index 19, ts=29*60000) starts bucket 2
    assert result[-1][0] == 29 * 60_000


def test_resample_partial_trailing_bucket_still_emitted():
    candles = make_1m_candles(10)  # less than one 29-minute bucket
    result = resample_candles(candles, 29)
    assert len(result) == 1
    assert result[0][4] == candles[-1][4]


def test_resample_39_49_59_minute_buckets_produce_expected_counts():
    for bucket in (39, 49, 59):
        candles = make_1m_candles(bucket * 3)
        result = resample_candles(candles, bucket)
        assert len(result) == 3
        assert result[0][0] == 0
        assert result[1][0] == bucket * 60_000
        assert result[2][0] == 2 * bucket * 60_000
