"""OMLX Drawdown Trading Advisor

Real-time advisor for profiting from drawdowns via immediate safety order placement.
Uses bounce probability analysis to decide when to add more position into dips.

Integration point for:
- Real-time dip detection
- Bounce probability analysis
- Safety order placement decisions
- Recovery monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from symbot_python.strategy.omlx_bounce_analyzer import (
    BounceAnalyzer,
    DipContext,
    BounceAnalysis
)

logger = logging.getLogger(__name__)


@dataclass
class DrawdownDecision:
    """Decision on whether to add safety order into dip.

    matched_patterns/dimension_scores/context carry the REAL computed
    bounce-analysis data through to callers that record trades for ML
    training (forward_omlx_tester.py) or calibration
    (dca_bot.py/dip_calibration_engine.py) — without these, both callers
    used to fall back to hardcoded stub values (dca_bot.py's own comment
    read "dimension_scores = {}  # Will be populated from analyzer", and
    forward_omlx_tester.py's read "volume_ratio: 1.2, # Would come from
    actual data" for six fields that never varied per trade). context is
    the full DipContext BounceAnalyzer scored — the raw market data (RSI,
    MACD, volume ratio, ATR, current drawdown %, etc.) every one of those
    stubbed fields has a real equivalent for.
    """
    should_add_safety: bool
    safety_amount: float  # As % of first order
    confidence: float  # 0-100
    reason: str
    time_estimate_candles: int
    liquidation_risk: str  # LOW/MEDIUM/HIGH
    # The raw BounceAnalysis.recommendation this decision was derived
    # from (ADD_SAFETY / ADD_SAFETY_AGGRESSIVE / WAIT_1_CANDLE / BAILOUT)
    # — callers that need the specific action, not just the boolean
    # should_add_safety, use this (e.g. dip_calibration_engine.py tracks
    # calibration per action-type).
    recommendation: str = ""
    # Optional/empty on the emergency-bailout path (dip_analysis_service.py
    # exits before a full BounceAnalyzer pass ever runs — RSI/MACD/etc.
    # for that instant were never computed, so there is no real context
    # to attach), always populated on a normal analyzed decision.
    matched_patterns: list[str] = field(default_factory=list)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    context: Optional[DipContext] = None


class OMLXDrawdownAdvisor:
    """Real-time drawdown profit advisor."""

    def __init__(self):
        self.bounce_analyzer = BounceAnalyzer()
        self.dip_start_time = None
        self.dip_first_score = None

    async def analyze_drawdown(
        self,
        entry_price: float,
        current_price: float,
        recent_high: float,
        recent_low: float,
        atr: float,
        atr_avg_20: float,
        rsi: float,
        macd_histogram: float,
        stochastic_k: float,
        current_volume: float,
        avg_volume_20: float,
        current_time_utc: int,
        day_of_week: int,
        bitcoin_trend: int,
        bid_ask_spread_bps: float,
        account_dd_percent: float,
        liquidation_buffer: float,
        bollinger_high: float,
        bollinger_low: float,
    ) -> DrawdownDecision:
        """Analyze current drawdown and decide on safety order placement."""

        dip_percent = abs((current_price - entry_price) / entry_price * 100)
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # Bollinger %B calculation
        range_width = bollinger_high - bollinger_low
        bollinger_pct_b = (
            (current_price - bollinger_low) / range_width
            if range_width > 0 else 0.5
        )

        # Check if wick formed (lower wick = rejection of lower prices)
        wick_formed = recent_low < current_price * 0.998

        # Check if candle closing up
        candle_closing_up = True  # Would need previous close, assume yes in dip

        # Build context
        context = DipContext(
            entry_price=entry_price,
            current_price=current_price,
            dip_percent=dip_percent,
            dip_candles=1,  # Would be updated per candle
            current_volume=current_volume,
            avg_volume_20=avg_volume_20,
            volume_ratio=volume_ratio,
            recent_high=recent_high,
            recent_low=recent_low,
            wick_formed=wick_formed,
            candle_closing_up=candle_closing_up,
            rsi=rsi,
            macd_histogram=macd_histogram,
            stochastic_k=stochastic_k,
            atr=atr,
            atr_avg_20=atr_avg_20,
            bollinger_high=bollinger_high,
            bollinger_low=bollinger_low,
            bollinger_pct_b=bollinger_pct_b,
            time_utc=current_time_utc,
            day_of_week=day_of_week,
            bitcoin_trend=bitcoin_trend,
            bid_ask_spread_bps=bid_ask_spread_bps,
            account_dd_percent=account_dd_percent,
            liquidation_buffer=liquidation_buffer,
        )

        # Analyze bounce probability
        analysis: BounceAnalysis = self.bounce_analyzer.analyze_dip(context)

        # Convert to safety order decision
        decision = self._make_decision(analysis, dip_percent, liquidation_buffer, context)

        return decision

    def _make_decision(
        self,
        analysis: BounceAnalysis,
        dip_percent: float,
        liquidation_buffer: float,
        context: DipContext,
    ) -> DrawdownDecision:
        """Convert bounce analysis into safety order decision."""

        probability = analysis.probability
        recommendation = analysis.recommendation

        # Determine if should add safety
        should_add_safety = recommendation in [
            "ADD_SAFETY",
            "ADD_SAFETY_AGGRESSIVE"
        ]

        # Determine safety amount based on dip and confidence
        if recommendation == "ADD_SAFETY_AGGRESSIVE":
            safety_amount = 50.0  # 50% of first order
        elif recommendation == "ADD_SAFETY":
            safety_amount = 50.0  # 50% of first order
        elif recommendation == "WAIT_1_CANDLE":
            safety_amount = 0.0
        else:  # BAILOUT
            safety_amount = 0.0

        # Liquidation risk assessment
        if liquidation_buffer < 10:
            liquidation_risk = "HIGH"
        elif liquidation_buffer < 15:
            liquidation_risk = "MEDIUM"
        else:
            liquidation_risk = "LOW"

        # Build reason
        reason = f"Bounce prob: {probability:.0f}% | Patterns: {', '.join(analysis.matched_patterns)}"

        return DrawdownDecision(
            should_add_safety=should_add_safety,
            safety_amount=safety_amount,
            confidence=probability,
            reason=reason,
            time_estimate_candles=int(analysis.time_to_tp_estimate),
            liquidation_risk=liquidation_risk,
            recommendation=recommendation,
            matched_patterns=analysis.matched_patterns,
            dimension_scores=analysis.dimension_scores,
            context=context,
        )

    def get_advice_string(self, decision: DrawdownDecision) -> str:
        """Format advice as readable string."""
        if decision.should_add_safety:
            action = "✅ ADD SAFETY ORDER"
        else:
            action = "⏸️  WAIT / BAILOUT"

        return (
            f"{action} | Conf: {decision.confidence:.0f}% | "
            f"Est: {decision.time_estimate_candles} candles | "
            f"Risk: {decision.liquidation_risk}\n"
            f"Reason: {decision.reason}"
        )


# Global advisor instance
_drawdown_advisor: Optional[OMLXDrawdownAdvisor] = None


async def get_drawdown_advisor() -> OMLXDrawdownAdvisor:
    """Get or create the global drawdown advisor."""
    global _drawdown_advisor
    if _drawdown_advisor is None:
        _drawdown_advisor = OMLXDrawdownAdvisor()
    return _drawdown_advisor
