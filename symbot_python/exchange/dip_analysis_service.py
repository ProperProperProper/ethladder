"""DIP Analysis Service - Integration layer for real-time bounce probability

Connects MarketDataCollector → OMLXDrawdownAdvisor to make safety order decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional
from datetime import datetime

from symbot_python.strategy.market_data_collector import (
    MarketDataCollector,
    OHLCV
)
from symbot_python.exchange.omlx_drawdown_advisor import (
    OMLXDrawdownAdvisor,
    DrawdownDecision
)

logger = logging.getLogger(__name__)


class DipAnalysisService:
    """Real-time dip detection and bounce analysis."""

    def __init__(self, emit_critical_alerts: bool = True):
        """Initialize service.

        emit_critical_alerts: whether a severe-crash dip logs at CRITICAL
        (which log_watcher turns into a native desktop notification) or
        just INFO. forward_omlx_tester.py constructs its OWN
        DipAnalysisService per test run (not the shared singleton
        dca_bot.py uses via get_dip_analysis_service()) and replays real
        historical candles continuously across many parameter
        combinations — hitting a real >3% historical move is routine
        there, not an emergency, and was firing a "SEVERE CRASH" native
        alert for ordinary backtest activity. Reserved for the live/paper
        trading engine's real position, where a severe crash is
        genuinely actionable.
        """
        self.data_collector = MarketDataCollector(window_size=50)
        self.advisor = OMLXDrawdownAdvisor()
        self.emit_critical_alerts = emit_critical_alerts

        # Tracking state
        self.entry_price: Optional[float] = None
        self.current_dip_price: Optional[float] = None
        self.dip_start_time: Optional[float] = None
        self.last_decision: Optional[DrawdownDecision] = None
        self.analysis_count = 0

    def update_with_candle(
        self,
        timestamp: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float
    ) -> None:
        """Feed new candle data into collector."""
        candle = OHLCV(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        self.data_collector.update(candle)

    def set_entry_price(self, price: float) -> None:
        """Set the entry price for monitoring dips."""
        self.entry_price = price
        self.current_dip_price = price
        self.dip_start_time = None
        logger.info(f"DIP_ANALYSIS | Entry price set to {price:.2f}")

    async def analyze_current_dip(
        self,
        current_price: float,
        account_balance: float,
        position_size: float,
        leverage: int = 11,
        tp_percent: float = 0.33,
        bid_ask_spread_bps: Optional[float] = None,
    ) -> Optional[DrawdownDecision]:
        """Analyze current position for bounce probability.

        bid_ask_spread_bps: real current spread in bps, when the caller
        has one (dca_bot.py's tick loop does — it already fetched a full
        Ticker with bid/ask this tick). Falls back to a fixed 2.0bps
        estimate when the caller has no live orderbook to read from
        (forward_omlx_tester.py replays OHLCV candle data only — candles
        carry no bid/ask, so there is no real spread to pass here; this
        is a genuine data-availability gap, not a wiring bug like the
        other previously-hardcoded fields were).

        Args:
            current_price: Current market price
            account_balance: Current account balance (for liquidation calc)
            position_size: Current position size
            leverage: Leverage used (default 11x)
            tp_percent: Take profit target as % (default 0.33%)

        Returns:
            DrawdownDecision or None if not in a dip
        """

        if self.entry_price is None or not self.data_collector.is_ready():
            return None

        # Calculate dip magnitude
        dip_percent = abs(
            (current_price - self.entry_price) / self.entry_price * 100
        )

        # Ignore tiny movements
        if dip_percent < 0.3:
            return None

        # Emergency bailout on severe crash
        if dip_percent > 3.0:
            log_fn = logger.critical if self.emit_critical_alerts else logger.info
            log_fn(
                f"SEVERE CRASH | Price: {current_price:.2f} | "
                f"DD: {dip_percent:.2f}% | BAILING OUT"
            )
            return DrawdownDecision(
                should_add_safety=False,
                safety_amount=0,
                confidence=0,
                reason="Severe crash >3%",
                time_estimate_candles=0,
                liquidation_risk="HIGH",
                recommendation="BAILOUT",
            )

        # Track dip start time
        if self.dip_start_time is None:
            self.dip_start_time = time.time()
            logger.info(f"DIP DETECTED | {dip_percent:.2f}% down | Starting analysis...")

        # Gather all market data
        rsi = self.data_collector.get_rsi() or 50
        macd_hist = self.data_collector.get_macd_histogram() or 0
        stoch_k = self.data_collector.get_stochastic_k() or 50
        atr = self.data_collector.get_atr() or 0
        atr_avg = self.data_collector.get_atr_average() or atr
        volume_current = self.data_collector.volumes[-1] if self.data_collector.volumes else 0
        volume_avg = self.data_collector.get_volume_average() or volume_current
        recent_high = self.data_collector.get_recent_high() or current_price
        recent_low = self.data_collector.get_recent_low() or current_price

        bb_bands = self.data_collector.get_bollinger_bands()
        if bb_bands:
            bb_high, _, bb_low = bb_bands
        else:
            bb_high = current_price * 1.02
            bb_low = current_price * 0.98

        bb_pct_b = self.data_collector.get_bollinger_pct_b() or 0.5
        volume_ratio = self.data_collector.get_volume_ratio() or 1.0
        bid_ask_spread = bid_ask_spread_bps if bid_ask_spread_bps is not None else 2.0

        # Calculate liquidation buffer
        account_dd = dip_percent * leverage
        liquidation_buffer = 100 - account_dd  # How much buffer before liquidation at 100%

        # Get current hour and day
        now = datetime.utcnow()
        current_hour = now.hour
        day_of_week = now.weekday()

        # Bitcoin trend (would integrate with external Bitcoin feed)
        bitcoin_trend = 0  # Placeholder: -1=down, 0=neutral, +1=up

        # Run analysis
        decision = await self.advisor.analyze_drawdown(
            entry_price=self.entry_price,
            current_price=current_price,
            recent_high=recent_high,
            recent_low=recent_low,
            atr=atr,
            atr_avg_20=atr_avg,
            rsi=rsi,
            macd_histogram=macd_hist,
            stochastic_k=stoch_k,
            current_volume=volume_current,
            avg_volume_20=volume_avg,
            current_time_utc=current_hour,
            day_of_week=day_of_week,
            bitcoin_trend=bitcoin_trend,
            bid_ask_spread_bps=bid_ask_spread,
            account_dd_percent=account_dd,
            liquidation_buffer=liquidation_buffer,
            bollinger_high=bb_high,
            bollinger_low=bb_low,
        )

        self.last_decision = decision
        self.analysis_count += 1

        # Log decision
        self._log_decision(
            dip_percent,
            current_price,
            decision,
            rsi,
            macd_hist,
            stoch_k
        )

        return decision

    def reset_dip_state(self) -> None:
        """Reset dip tracking when dip resolves."""
        self.dip_start_time = None
        self.current_dip_price = None
        logger.info("DIP_ANALYSIS | Dip resolved, reset tracking")

    def _log_decision(
        self,
        dip_percent: float,
        current_price: float,
        decision: DrawdownDecision,
        rsi: float,
        macd: float,
        stoch: float
    ) -> None:
        """Log analysis decision."""

        if decision.should_add_safety:
            action_emoji = "✅"
            action_text = "ADD SAFETY"
        else:
            action_emoji = "⏸️ "
            action_text = "WAIT/BAILOUT"

        logger.info(
            f"{action_emoji} DIP ANALYSIS | "
            f"Price: {current_price:.2f} | "
            f"DD: {dip_percent:.2f}% | "
            f"RSI: {rsi:.0f} | "
            f"MACD: {macd:.3f} | "
            f"Stoch: {stoch:.0f} | "
            f"Bounce Prob: {decision.confidence:.0f}% | "
            f"Action: {action_text}"
        )

        # Log detailed reason
        logger.info(f"   Reason: {decision.reason}")
        logger.info(f"   Est Time to TP: {decision.time_estimate_candles} candles")
        logger.info(f"   Liquidation Risk: {decision.liquidation_risk}")

    def get_status(self) -> dict:
        """Get current service status."""
        return {
            "is_ready": self.data_collector.is_ready(),
            "candles_collected": len(self.data_collector.closes),
            "entry_price": self.entry_price,
            "in_dip": self.dip_start_time is not None,
            "dip_duration_seconds": (
                time.time() - self.dip_start_time
                if self.dip_start_time else None
            ),
            "analysis_count": self.analysis_count,
            "last_decision": self.last_decision,
            "data_status": self.data_collector.get_status()
        }


# Global service instance
_dip_service: Optional[DipAnalysisService] = None


async def get_dip_analysis_service() -> DipAnalysisService:
    """Get or create the global DIP analysis service."""
    global _dip_service
    if _dip_service is None:
        _dip_service = DipAnalysisService()
    return _dip_service
