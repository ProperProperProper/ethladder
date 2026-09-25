"""Candle aggregation helpers.

Bybit's v5 kline intervals are fixed (1,3,5,15,30,60,120,240,360,720,D,W,M)
and a single request caps out at 1000 rows. Two things this module adds
on top of that:

- resample_candles(): build ARBITRARY minute-width bars (e.g. 29m, 39m)
  by aggregating 1-minute candles — pure, no network, fully testable.
- Pagination lives in api/app.py (it needs a live session), but the
  function signature there is built around this module's Candle shape.
"""

from __future__ import annotations

import math

Candle = list[float]  # [ts_ms, open, high, low, close, volume]

BYBIT_NATIVE_INTERVALS = {
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M",
}

DEFAULT_BACKTEST_DAYS = 14  # locked in: every backtest/walk-forward run spans this period


def is_native_interval(interval: str) -> bool:
    return interval in BYBIT_NATIVE_INTERVALS


def interval_minutes(interval: str) -> float:
    """Width of one candle at this interval, in minutes. Covers native
    Bybit intervals (numeric-minute strings plus D/W/M) and custom
    aggregated widths (e.g. "29", "39"), which are already plain minutes.
    """
    if interval == "D":
        return 1440.0
    if interval == "W":
        return 10080.0
    if interval == "M":
        return 43200.0  # approx: 30 days
    return float(interval)


def bars_for_days(interval: str, days: float = DEFAULT_BACKTEST_DAYS) -> int:
    """How many candles at `interval` are needed to span `days` of history."""
    minutes = interval_minutes(interval)
    return max(1, math.ceil(days * 24 * 60 / minutes))


def resample_candles(candles: list[Candle], bucket_minutes: int) -> list[Candle]:
    """Aggregate 1-minute-spaced OHLCV candles into fixed bucket_minutes-
    wide bars. Buckets are aligned to fixed-size windows from the UNIX
    epoch (UTC), not to the first candle's own timestamp, so the result
    is stable no matter where the fetch window happened to start.

    Input must be sorted oldest-first. A bucket that only received a
    partial run of 1-minute candles at either edge of the input (i.e. the
    very first or very last bucket) is still emitted — callers doing a
    live/incomplete-bar-sensitive backtest should drop the last bucket if
    it isn't yet closed.
    """
    if bucket_minutes <= 0 or not candles:
        return list(candles)

    bucket_ms = bucket_minutes * 60_000
    order: list[int] = []
    buckets: dict[int, list[Candle]] = {}
    for c in candles:
        ts = int(c[0])
        bucket_start = (ts // bucket_ms) * bucket_ms
        group = buckets.get(bucket_start)
        if group is None:
            group = []
            buckets[bucket_start] = group
            order.append(bucket_start)
        group.append(c)

    result: list[Candle] = []
    for bucket_start in order:
        group = buckets[bucket_start]
        open_price = group[0][1]
        high = max(row[2] for row in group)
        low = min(row[3] for row in group)
        close = group[-1][4]
        volume = sum(row[5] for row in group)
        result.append([float(bucket_start), open_price, high, low, close, volume])
    return result
