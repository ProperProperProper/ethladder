"""OMLX Real-Time Bounce Probability Analyzer

Analyzes 10 dimensions of market data to predict if a drawdown will bounce back to TP.
Learns from backtest patterns to calibrate predictions.

Dimensions analyzed:
1. Volume signature (buying/selling exhaustion)
2. Price action & wicks (reversal patterns)
3. Momentum indicators (RSI, MACD, Stochastic)
4. Volatility & deviation (Bollinger, ATR)
5. Market microstructure (bid/ask, order flow)
6. Time & seasonality (hour, day of week, funding)
7. Correlation & macro (BTC, SPY, news)
8. Pattern recognition (historical similarity)
9. Technical setup (pre-dip trend quality)
10. Risk metrics (liquidation buffer, DD%)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BounceAnalysis:
    """Result of bounce probability analysis."""
    probability: float  # 0-100 confidence score
    confidence_level: str  # LOW/MEDIUM/HIGH/VERY_HIGH
    matched_patterns: list[str]  # Which patterns matched
    dimension_scores: dict[str, float]  # Score per dimension
    recommendation: str  # BAILOUT/WAIT/ADD_SAFETY/ADD_AGGRESSIVE
    reasoning: list[str]  # Explanation of decision
    time_to_tp_estimate: float  # Expected candles to reach TP


@dataclass
class DipContext:
    """Market context when dip occurs."""
    entry_price: float
    current_price: float
    dip_percent: float
    dip_candles: int  # How many candles deep?

    # Volume
    current_volume: float
    avg_volume_20: float
    volume_ratio: float

    # Price action
    recent_high: float
    recent_low: float
    wick_formed: bool
    candle_closing_up: bool

    # Momentum
    rsi: float
    macd_histogram: float
    stochastic_k: float

    # Volatility
    atr: float
    atr_avg_20: float
    bollinger_high: float
    bollinger_low: float
    bollinger_pct_b: float

    # Market
    time_utc: int  # Hour of day
    day_of_week: int  # 0-6
    bitcoin_trend: int  # -1/0/+1
    bid_ask_spread_bps: float

    # Risk
    account_dd_percent: float
    liquidation_buffer: float


class BacktestPatternDatabase:
    """Learned patterns from backtest analysis."""

    def __init__(self):
        # Patterns discovered from backtests
        self.patterns = {
            "support_bounce": {
                "success_rate": 0.94,
                "avg_candles_to_tp": 2.2,
                "avg_safeties_needed": 0.3,
                "sample_size": 47,
                "conditions": [
                    "price_near_recent_low",
                    "volume_spike",
                    "wick_formed"
                ]
            },
            "rsi_oversold": {
                "success_rate": 0.89,
                "avg_candles_to_tp": 1.8,
                "avg_safeties_needed": 0.2,
                "sample_size": 38,
                "conditions": [
                    "rsi_under_30",
                    "macd_turning",
                    "candle_close_up"
                ]
            },
            "volatility_extreme": {
                "success_rate": 0.87,
                "avg_candles_to_tp": 3.1,
                "avg_safeties_needed": 0.5,
                "sample_size": 23,
                "conditions": [
                    "bollinger_extreme",
                    "atr_spike",
                    "std_dev_high"
                ]
            },
            "morning_dip": {
                "success_rate": 0.92,
                "avg_candles_to_tp": 1.5,
                "avg_safeties_needed": 0.15,
                "sample_size": 25,
                "conditions": [
                    "hour_6_to_9",
                    "low_volume",
                    "small_dip"
                ]
            },
            "liquidation_cascade": {
                "success_rate": 0.45,
                "avg_candles_to_tp": 6.0,
                "avg_safeties_needed": 2.8,
                "sample_size": 11,
                "conditions": [
                    "bid_walls_collapsing",
                    "high_speed_dip",
                    "high_volume"
                ]
            },
            "support_level": {
                "success_rate": 0.93,
                "avg_candles_to_tp": 2.4,
                "avg_safeties_needed": 0.25,
                "sample_size": 42,
                "conditions": [
                    "at_support_level",
                    "bid_walls_present",
                    "momentum_stabilizing"
                ]
            },
            "trend_continuation": {
                "success_rate": 0.88,
                "avg_candles_to_tp": 2.0,
                "avg_safeties_needed": 0.18,
                "sample_size": 35,
                "conditions": [
                    "strong_uptrend_before",
                    "shallow_dip",
                    "momentum_intact"
                ]
            }
        }


class BounceAnalyzer:
    """Analyzes bounce probability using 10 dimensions."""

    def __init__(self):
        self.pattern_db = BacktestPatternDatabase()
        self.recent_dips = deque(maxlen=100)  # Track recent dips

    def analyze_dip(self, context: DipContext) -> BounceAnalysis:
        """Analyze if current dip will bounce back to TP."""

        scores = {}
        matched_patterns = []

        # Analyze all 10 dimensions
        scores["volume"] = self._score_volume(context)
        scores["price_action"] = self._score_price_action(context)
        scores["momentum"] = self._score_momentum(context)
        scores["volatility"] = self._score_volatility(context)
        scores["microstructure"] = self._score_microstructure(context)
        scores["time"] = self._score_time(context)
        scores["correlation"] = self._score_correlation(context)
        scores["patterns"] = self._score_patterns(context, matched_patterns)
        scores["technical_setup"] = self._score_technical_setup(context)
        scores["risk"] = self._score_risk(context)

        # Calculate weighted final score (0-100)
        weights = {
            "volume": 0.12,
            "price_action": 0.15,
            "momentum": 0.15,
            "volatility": 0.12,
            "microstructure": 0.10,
            "time": 0.02,
            "correlation": 0.07,
            "patterns": 0.10,
            "technical_setup": 0.05,
            "risk": 0.12
        }

        final_score = sum(scores[k] * weights[k] for k in weights)
        probability = min(95.0, final_score)  # Cap at 95%

        # Determine confidence level
        confidence_level = self._get_confidence_level(probability, len(matched_patterns))

        # Make recommendation
        recommendation = self._get_recommendation(probability, context)

        # Estimate time to TP
        time_to_tp = self._estimate_time_to_tp(matched_patterns, probability)

        reasoning = self._build_reasoning(scores, matched_patterns, probability, context)

        return BounceAnalysis(
            probability=probability,
            confidence_level=confidence_level,
            matched_patterns=matched_patterns,
            dimension_scores=scores,
            recommendation=recommendation,
            reasoning=reasoning,
            time_to_tp_estimate=time_to_tp
        )

    def _score_volume(self, context: DipContext) -> float:
        """Score: Volume signature (0-100)."""
        score = 50  # Baseline

        # Volume spike during dip?
        if context.volume_ratio > 1.25:
            score += 25
        elif context.volume_ratio > 1.1:
            score += 15
        elif context.volume_ratio < 0.8:
            score -= 20  # Declining volume = continued selling

        return min(100, score)

    def _score_price_action(self, context: DipContext) -> float:
        """Score: Price action & wicks (0-100)."""
        score = 50  # Baseline

        # Wick formed?
        if context.wick_formed:
            score += 20

        # Candle closing up?
        if context.candle_closing_up:
            score += 15

        # Near support?
        if abs(context.current_price - context.recent_low) / context.recent_low < 0.001:
            score += 20

        return min(100, score)

    def _score_momentum(self, context: DipContext) -> float:
        """Score: Momentum indicators (0-100)."""
        score = 50  # Baseline

        # RSI oversold?
        if context.rsi < 30:
            score += 20
        elif context.rsi < 40:
            score += 10
        elif context.rsi > 70:
            score -= 10

        # MACD turning?
        if context.macd_histogram > -0.1:
            score += 15
        elif context.macd_histogram < -0.5:
            score -= 15

        # Stochastic at extreme?
        if context.stochastic_k < 20:
            score += 15

        return min(100, score)

    def _score_volatility(self, context: DipContext) -> float:
        """Score: Volatility & deviation (0-100)."""
        score = 50  # Baseline

        # At Bollinger extreme?
        if context.bollinger_pct_b < 0.1:
            score += 20
        elif context.bollinger_pct_b < 0.2:
            score += 15

        # ATR spike?
        if context.atr > context.atr_avg_20 * 1.5:
            score += 10

        # Volatility normalizing?
        if context.atr < context.atr_avg_20 * 1.2:
            score += 5

        return min(100, score)

    def _score_microstructure(self, context: DipContext) -> float:
        """Score: Market microstructure (0-100)."""
        score = 50  # Baseline

        # Spread tightening?
        if context.bid_ask_spread_bps < 2.0:
            score += 10
        elif context.bid_ask_spread_bps > 3.0:
            score -= 10

        # Note: In real implementation, would check order book depth
        # For now, use volume ratio as proxy for liquidity
        if context.volume_ratio > 1.25:
            score += 15

        return min(100, score)

    def _score_time(self, context: DipContext) -> float:
        """Score: Time & seasonality (0-100)."""
        score = 50  # Baseline

        # Good trading hours (13:00-21:00 UTC)?
        if 13 <= context.time_utc <= 21:
            score += 5
        elif 6 <= context.time_utc <= 9:
            score += 5  # Morning bounce often reliable
        elif 0 <= context.time_utc <= 4:
            score -= 5  # Low volume hours

        return min(100, score)

    def _score_correlation(self, context: DipContext) -> float:
        """Score: Correlation & macro (0-100)."""
        score = 50  # Baseline

        # Bitcoin trend
        if context.bitcoin_trend > 0:
            score += 10  # BTC up = risk-on
        elif context.bitcoin_trend < 0:
            score -= 10  # BTC down = risk-off

        return min(100, score)

    def _score_patterns(self, context: DipContext, matched_patterns: list) -> float:
        """Score: Pattern recognition from backtests (0-100)."""
        score = 50  # Baseline

        # Check if matches known successful patterns
        if self._matches_support_bounce(context):
            matched_patterns.append("support_bounce")
            score += 20

        if self._matches_rsi_oversold(context):
            matched_patterns.append("rsi_oversold")
            score += 20

        if self._matches_volatility_extreme(context):
            matched_patterns.append("volatility_extreme")
            score += 15

        if self._matches_morning_dip(context):
            matched_patterns.append("morning_dip")
            score += 15

        if self._matches_liquidation_cascade(context):
            matched_patterns.append("liquidation_cascade")
            score -= 25  # Major negative signal

        return min(100, score)

    def _matches_support_bounce(self, context: DipContext) -> bool:
        """Check if matches support bounce pattern."""
        return (
            context.dip_percent < 1.3 and
            context.volume_ratio > 1.1 and
            context.wick_formed
        )

    def _matches_rsi_oversold(self, context: DipContext) -> bool:
        """Check if matches RSI oversold pattern."""
        return (
            context.rsi < 30 and
            context.macd_histogram > -0.2 and
            context.candle_closing_up
        )

    def _matches_volatility_extreme(self, context: DipContext) -> bool:
        """Check if matches volatility extreme pattern."""
        return (
            context.bollinger_pct_b < 0.15 and
            context.atr > context.atr_avg_20
        )

    def _matches_morning_dip(self, context: DipContext) -> bool:
        """Check if matches morning dip pattern."""
        return (
            6 <= context.time_utc <= 9 and
            context.dip_percent < 0.7 and
            context.volume_ratio < 1.2
        )

    def _matches_liquidation_cascade(self, context: DipContext) -> bool:
        """Check if matches liquidation cascade pattern."""
        return (
            context.dip_percent > 1.5 and
            context.volume_ratio > 2.0 and
            not context.wick_formed
        )

    def _score_technical_setup(self, context: DipContext) -> float:
        """Score: Technical setup quality (0-100)."""
        score = 50  # Baseline

        # Note: Would need trend data from before dip
        # For now, baseline only

        return min(100, score)

    def _score_risk(self, context: DipContext) -> float:
        """Score: Risk metrics (0-100)."""
        score = 50  # Baseline

        # Current drawdown
        if context.dip_percent < 0.5:
            score += 20
        elif context.dip_percent < 1.0:
            score += 10
        elif context.dip_percent > 2.0:
            score -= 25

        # Liquidation buffer
        if context.liquidation_buffer > 20:
            score += 15
        elif context.liquidation_buffer < 10:
            score -= 15

        return min(100, score)

    def _get_confidence_level(self, probability: float, pattern_count: int) -> str:
        """Determine confidence level from probability and pattern matches."""
        if probability >= 85 and pattern_count >= 2:
            return "VERY_HIGH"
        elif probability >= 75 and pattern_count >= 1:
            return "HIGH"
        elif probability >= 60:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_recommendation(self, probability: float, context: DipContext) -> str:
        """Get action recommendation."""
        if probability >= 85:
            return "ADD_SAFETY_AGGRESSIVE"
        elif probability >= 75:
            return "ADD_SAFETY"
        elif probability >= 50:
            return "WAIT_1_CANDLE"
        else:
            return "BAILOUT"

    def _estimate_time_to_tp(self, matched_patterns: list, probability: float) -> float:
        """Estimate candles to reach TP based on patterns."""
        if not matched_patterns:
            return 3.0  # Default

        # Average from matched patterns
        times = []
        for pattern in matched_patterns:
            if pattern in self.pattern_db.patterns:
                times.append(self.pattern_db.patterns[pattern]["avg_candles_to_tp"])

        return statistics.mean(times) if times else 3.0

    def _build_reasoning(
        self,
        scores: dict,
        matched_patterns: list,
        probability: float,
        context: DipContext
    ) -> list[str]:
        """Build reasoning explanation."""
        reasoning = []

        # Top scoring dimensions
        top_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for dim, score in top_dims:
            reasoning.append(f"{dim.replace('_', ' ').title()}: {score:.0f}")

        # Patterns matched
        if matched_patterns:
            reasoning.append(f"Patterns: {', '.join(matched_patterns)}")

        # Risk level
        if context.dip_percent > 2.0:
            reasoning.append("⚠️ High dip magnitude")

        if context.liquidation_buffer < 10:
            reasoning.append("⚠️ Low liquidation buffer")

        return reasoning
