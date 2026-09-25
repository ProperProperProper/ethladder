"""Market Data Collector for OMLX Drawdown Advisor

Maintains rolling windows of market data and calculates technical indicators
needed for bounce probability analysis.

Calculated indicators:
- RSI(14)
- MACD(12,26,9)
- Stochastic K(14,3)
- ATR(14)
- Bollinger Bands(20,2)
- Volume averages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Single candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataCollector:
    """Collect and maintain technical indicator state."""

    def __init__(self, window_size: int = 50):
        """Initialize with rolling windows."""
        self.window_size = window_size

        # Price windows
        self.closes = deque(maxlen=window_size)
        self.highs = deque(maxlen=window_size)
        self.lows = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.candles = deque(maxlen=window_size)

        # Indicator buffers
        self.rsi_gains = deque(maxlen=14)
        self.rsi_losses = deque(maxlen=14)
        self.ema12 = None  # MACD fast
        self.ema26 = None  # MACD slow
        self.signal_line = None  # MACD signal
        self.macd_histogram_values = deque(maxlen=9)

    def update(self, candle: OHLCV) -> None:
        """Update with new candle data."""
        self.candles.append(candle)
        self.closes.append(candle.close)
        self.highs.append(candle.high)
        self.lows.append(candle.low)
        self.volumes.append(candle.volume)

        # Update indicators
        self._update_rsi(candle)
        self._update_macd(candle)

    def _update_rsi(self, candle: OHLCV) -> None:
        """Update RSI(14) calculation."""
        if len(self.closes) < 2:
            return

        change = candle.close - self.closes[-2]
        if change > 0:
            self.rsi_gains.append(change)
            self.rsi_losses.append(0)
        else:
            self.rsi_gains.append(0)
            self.rsi_losses.append(abs(change))

    def _update_macd(self, candle: OHLCV) -> None:
        """Update MACD calculation."""
        if len(self.closes) < 26:
            return

        # Calculate or update EMAs
        if self.ema12 is None:
            self.ema12 = statistics.mean(list(self.closes)[-12:])
            self.ema26 = statistics.mean(list(self.closes)[-26:])
        else:
            alpha_12 = 2 / (12 + 1)
            alpha_26 = 2 / (26 + 1)
            self.ema12 = candle.close * alpha_12 + self.ema12 * (1 - alpha_12)
            self.ema26 = candle.close * alpha_26 + self.ema26 * (1 - alpha_26)

        macd_line = self.ema12 - self.ema26

        # Update signal line (9-period EMA of MACD)
        if self.signal_line is None and len(self.closes) >= 34:
            self.signal_line = statistics.mean(
                [self.ema12 - self.ema26 for _ in range(9)]
            )
        elif self.signal_line is not None:
            alpha_9 = 2 / (9 + 1)
            self.signal_line = (
                macd_line * alpha_9 + self.signal_line * (1 - alpha_9)
            )

        # Store histogram
        if self.signal_line is not None:
            self.macd_histogram_values.append(macd_line - self.signal_line)

    # --- Getters for current indicator values ---

    def get_rsi(self, period: int = 14) -> float | None:
        """Get current RSI(14)."""
        if len(self.rsi_gains) < period:
            return None

        avg_gain = statistics.mean(list(self.rsi_gains)[-period:])
        avg_loss = statistics.mean(list(self.rsi_losses)[-period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_macd_histogram(self) -> float | None:
        """Get current MACD histogram."""
        if not self.macd_histogram_values:
            return None
        return self.macd_histogram_values[-1]

    def get_stochastic_k(self, period: int = 14) -> float | None:
        """Get Stochastic K (14 period)."""
        if len(self.lows) < period:
            return None

        recent_lows = list(self.lows)[-period:]
        recent_highs = list(self.highs)[-period:]

        low = min(recent_lows)
        high = max(recent_highs)

        if high == low:
            return 50.0

        current_close = self.closes[-1]
        k = ((current_close - low) / (high - low)) * 100
        return k

    def get_atr(self, period: int = 14) -> float | None:
        """Get ATR(14)."""
        if len(self.candles) < period:
            return None

        trs = []
        for i in range(max(0, len(self.candles) - period), len(self.candles)):
            candle = self.candles[i]
            if i == 0:
                tr = candle.high - candle.low
            else:
                prev_close = self.candles[i - 1].close
                tr = max(
                    candle.high - candle.low,
                    abs(candle.high - prev_close),
                    abs(candle.low - prev_close)
                )
            trs.append(tr)

        return statistics.mean(trs) if trs else None

    def get_atr_average(self, period: int = 14, avg_period: int = 20) -> float | None:
        """Get average ATR over last avg_period candles."""
        if len(self.candles) < max(period, avg_period):
            return None

        atrs = []
        for i in range(len(self.candles) - avg_period + 1, len(self.candles)):
            window_start = max(0, i - period + 1)
            trs = []
            for j in range(window_start, i + 1):
                candle = self.candles[j]
                if j == 0:
                    tr = candle.high - candle.low
                else:
                    prev_close = self.candles[j - 1].close
                    tr = max(
                        candle.high - candle.low,
                        abs(candle.high - prev_close),
                        abs(candle.low - prev_close)
                    )
                trs.append(tr)

            if trs:
                atrs.append(statistics.mean(trs))

        return statistics.mean(atrs) if atrs else None

    def get_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float] | None:
        """Get Bollinger Bands (20,2)."""
        if len(self.closes) < period:
            return None

        recent_closes = list(self.closes)[-period:]
        sma = statistics.mean(recent_closes)
        stdev = statistics.stdev(recent_closes)

        bb_high = sma + (stdev * std_dev)
        bb_low = sma - (stdev * std_dev)

        return bb_high, sma, bb_low

    def get_bollinger_pct_b(self, period: int = 20, std_dev: float = 2.0) -> float | None:
        """Get Bollinger %B indicator."""
        bb = self.get_bollinger_bands(period, std_dev)
        if not bb:
            return None

        bb_high, _, bb_low = bb
        current_price = self.closes[-1]
        bb_range = bb_high - bb_low

        if bb_range == 0:
            return 0.5

        pct_b = (current_price - bb_low) / bb_range
        return max(0, min(1, pct_b))

    def get_volume_ratio(self, period: int = 20) -> float | None:
        """Get ratio of current volume to average."""
        if len(self.volumes) < period:
            return None

        recent_volumes = list(self.volumes)[-period:]
        avg_volume = statistics.mean(recent_volumes)

        if avg_volume == 0:
            return 1.0

        return self.volumes[-1] / avg_volume

    def get_recent_high(self, period: int = 20) -> float | None:
        """Get highest high in last N candles."""
        if len(self.highs) < period:
            return None
        return max(list(self.highs)[-period:])

    def get_recent_low(self, period: int = 20) -> float | None:
        """Get lowest low in last N candles."""
        if len(self.lows) < period:
            return None
        return min(list(self.lows)[-period:])

    def get_volatility_std_dev(self, period: int = 20) -> float | None:
        """Get standard deviation of closes."""
        if len(self.closes) < period:
            return None

        recent_closes = list(self.closes)[-period:]
        return statistics.stdev(recent_closes) if len(recent_closes) > 1 else None

    def get_volume_average(self, period: int = 20) -> float | None:
        """Get average volume over period."""
        if len(self.volumes) < period:
            return None

        recent_volumes = list(self.volumes)[-period:]
        return statistics.mean(recent_volumes)

    def has_wick_formed(self, wick_threshold: float = 0.002) -> bool:
        """Check if current candle has lower wick (rejection of lower prices)."""
        if not self.candles:
            return False

        candle = self.candles[-1]
        # Wick if low is below close by threshold
        return (candle.close - candle.low) / candle.close > wick_threshold

    def is_candle_closing_up(self) -> bool | None:
        """Check if current candle closing above open."""
        if not self.candles:
            return None

        candle = self.candles[-1]
        return candle.close > candle.open

    def get_candle_close_vs_open_ratio(self) -> float | None:
        """Get how much candle closed up from open as ratio."""
        if not self.candles:
            return None

        candle = self.candles[-1]
        if candle.open == 0:
            return 0.0

        return (candle.close - candle.open) / candle.open

    # --- Helper methods ---

    def is_ready(self, min_candles: int = 50) -> bool:
        """Check if we have enough data for analysis."""
        return len(self.closes) >= min_candles

    def get_status(self) -> dict:
        """Get current status of all indicators."""
        return {
            "candles_collected": len(self.closes),
            "rsi": self.get_rsi(),
            "macd_histogram": self.get_macd_histogram(),
            "stochastic_k": self.get_stochastic_k(),
            "atr": self.get_atr(),
            "atr_avg_20": self.get_atr_average(),
            "volume_ratio": self.get_volume_ratio(),
            "recent_high": self.get_recent_high(),
            "recent_low": self.get_recent_low(),
            "volatility_std_dev": self.get_volatility_std_dev(),
            "volume_average": self.get_volume_average(),
            "has_wick": self.has_wick_formed(),
            "closing_up": self.is_candle_closing_up(),
        }
