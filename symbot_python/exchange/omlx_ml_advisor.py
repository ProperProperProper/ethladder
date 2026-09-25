"""OMLX ML Advisor - Integrates ML models into OMLX decision making.

🔴 CRITICAL: BYBIT MAINNET ONLY
This advisor is trained on BYBIT MAINNET data exclusively.
- Never testnet (testnet = fake, worthless)
- Never synthetic data
- Never other exchanges
If data source changes, all models become invalid.

Combines:
- XGBoost outcome predictor (probability of win)
- RL decision optimizer (optimal action)
- Original OMLX scoring (market dimensions)

Produces higher confidence, more accurate decisions.
"""

import logging
from typing import Optional, Tuple

from symbot_python.ml.outcome_predictor import OutcomePredictor, TradeFeatures
from symbot_python.ml.rl_decision_optimizer import (
    RLDecisionOptimizer, state_from_market_data
)

logger = logging.getLogger(__name__)


class OMLXMLAdvisor:
    """ML-enhanced OMLX advisor combining multiple decision models."""

    def __init__(self):
        """Initialize ML advisor with loaded models."""
        self.outcome_predictor = OutcomePredictor()
        self.rl_optimizer = RLDecisionOptimizer()
        self.models_available = self._check_models()

    def _check_models(self) -> bool:
        """Check if ML models are available."""
        has_outcome = self.outcome_predictor.model is not None
        has_rl = len(self.rl_optimizer.Q_table) > 0

        if has_outcome:
            logger.info("✓ Outcome predictor available")
        if has_rl:
            logger.info(f"✓ RL optimizer available ({len(self.rl_optimizer.Q_table)} states)")

        return has_outcome or has_rl

    def enhance_decision(
        self,
        omlx_decision: str,  # Original OMLX decision: ADD_SAFETY, WAIT, BAILOUT
        omlx_confidence: float,  # Original OMLX confidence 0-100
        trade_features: dict,  # Market data and trade state
    ) -> Tuple[str, float]:
        """Enhance OMLX decision with ML models.

        Returns:
            (final_decision, final_confidence)
        """
        if not self.models_available:
            # No ML models, return original decision
            return omlx_decision, omlx_confidence

        try:
            ml_decision = omlx_decision
            ml_confidence = omlx_confidence

            # Get ML predictions
            outcome_prob = self._get_outcome_probability(trade_features)
            rl_decision, rl_confidence = self._get_rl_decision(trade_features)

            # Combine decisions
            ml_decision, ml_confidence = self._combine_decisions(
                omlx_decision, omlx_confidence,
                outcome_prob,
                rl_decision, rl_confidence
            )

            return ml_decision, ml_confidence
        except Exception as e:
            logger.warning(f"ML enhancement failed: {e}, using original decision")
            return omlx_decision, omlx_confidence

    def _get_outcome_probability(self, trade_features: dict) -> float:
        """Get win probability from outcome predictor."""
        if self.outcome_predictor.model is None:
            return 0.5  # No model, neutral

        try:
            features = TradeFeatures(
                volume_ratio=trade_features.get('volume_ratio', 1.0),
                price_change_pct=trade_features.get('price_change_pct', 0.0),
                momentum_score=trade_features.get('momentum_score', 50.0),
                volatility_pct=trade_features.get('volatility_pct', 1.0),
                spread_bps=trade_features.get('spread_bps', 10.0),
                pattern_success_rate=trade_features.get('pattern_success_rate', 50.0),
                dip_depth_pct=trade_features.get('dip_depth_pct', 0.5),
                entry_confidence=trade_features.get('entry_confidence', 50.0),
                leverage=trade_features.get('leverage', 11),
                candles_since_entry=trade_features.get('candles_since_entry', 0),
                current_loss_pct=trade_features.get('current_loss_pct', 0.0)
            )

            result = self.outcome_predictor.predict(features)
            return result.win_probability * 100  # Convert to 0-100
        except Exception as e:
            logger.warning(f"Outcome prediction error: {e}")
            return 50.0

    def _get_rl_decision(self, trade_features: dict) -> Tuple[str, float]:
        """Get decision from RL optimizer."""
        if len(self.rl_optimizer.Q_table) == 0:
            return 'WAIT', 0.0  # No RL model

        try:
            state = state_from_market_data(trade_features)
            action, confidence = self.rl_optimizer.get_recommendation(state)
            return action, confidence
        except Exception as e:
            logger.warning(f"RL decision error: {e}")
            return 'WAIT', 0.0

    def _combine_decisions(
        self,
        omlx_decision: str,
        omlx_confidence: float,
        outcome_prob: float,
        rl_decision: str,
        rl_confidence: float
    ) -> Tuple[str, float]:
        """Combine all decisions into final decision.

        Logic:
        1. If outcome probability is very high (>85%), boost confidence in ADD_SAFETY
        2. If outcome probability is very low (<35%), recommend BAILOUT
        3. If RL and OMLX agree, boost confidence
        4. Average confidences
        """

        # Decision adjustment based on outcome probability
        if outcome_prob > 85:
            # Very likely to win, prefer ADD_SAFETY
            decision = 'ADD_SAFETY'
            outcome_confidence = min(100, outcome_prob)
        elif outcome_prob < 35:
            # Very likely to lose, prefer BAILOUT
            decision = 'BAILOUT'
            outcome_confidence = min(100, 100 - outcome_prob)
        else:
            # Use RL decision if available, otherwise OMLX
            decision = rl_decision if rl_confidence > 30 else omlx_decision
            outcome_confidence = 50

        # Agreement bonus
        if decision == omlx_decision:
            agreement_bonus = 10
        else:
            agreement_bonus = -5

        # Combine confidences with weights
        final_confidence = (
            omlx_confidence * 0.4 +  # Original OMLX
            outcome_confidence * 0.35 +  # Outcome predictor
            rl_confidence * 0.25  # RL optimizer
        )
        final_confidence = max(0, min(100, final_confidence + agreement_bonus))

        return decision, final_confidence


def get_ml_advisor() -> OMLXMLAdvisor:
    """Get singleton ML advisor instance."""
    if not hasattr(get_ml_advisor, '_instance'):
        get_ml_advisor._instance = OMLXMLAdvisor()
    return get_ml_advisor._instance
