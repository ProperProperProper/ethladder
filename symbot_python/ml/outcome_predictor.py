"""XGBoost Outcome Predictor - Predicts WIN/LOSS probability for trades.

Trained on historical trade data to predict if a trade will be profitable.
Used by OMLX to adjust confidence scores.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
import pickle
import numpy as np
from dataclasses import dataclass, asdict

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TradeFeatures:
    """Features extracted from a trade for prediction."""
    # Market dimensions
    volume_ratio: float  # Current volume / avg volume
    price_change_pct: float  # Price change %
    momentum_score: float  # RSI-like momentum 0-100
    volatility_pct: float  # Current volatility %
    spread_bps: float  # Bid-ask spread in bps

    # Pattern
    pattern_success_rate: float  # Historical success rate of pattern 0-100

    # Entry conditions
    dip_depth_pct: float  # How deep the dip was
    entry_confidence: float  # OMLX confidence 0-100
    leverage: int  # Leverage used (11x)

    # Position state
    candles_since_entry: int  # How many candles since entry
    current_loss_pct: float  # Current unrealized loss %


@dataclass
class PredictionResult:
    """Result of outcome prediction."""
    win_probability: float  # 0.0-1.0
    loss_probability: float  # 0.0-1.0
    confidence: float  # Model confidence in prediction
    recommendation: str  # ADD_SAFETY, WAIT, BAILOUT


class OutcomePredictor:
    """XGBoost model for predicting trade outcomes."""

    def __init__(self, model_path: str = 'omlx_outcome_model.pkl'):
        """Initialize predictor, load existing model or create new."""
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = [
            'volume_ratio', 'price_change_pct', 'momentum_score', 'volatility_pct',
            'spread_bps', 'pattern_success_rate', 'dip_depth_pct', 'entry_confidence',
            'leverage', 'candles_since_entry', 'current_loss_pct'
        ]

        if self.model_path.exists():
            self._load_model()
        else:
            logger.info("No model found, will train on first batch of data")

    def _load_model(self):
        """Load model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✓ Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def save_model(self):
        """Save model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        try:
            # Atomic write (temp file + os.replace) — same reasoning as
            # rl_decision_optimizer.py's save_model(): this .pkl is
            # reloaded live by omlx_ml_advisor.py on a different
            # schedule than this one saves on, and a plain in-place
            # write leaves a real window for a torn (unpickleable) read.
            tmp_path = f"{self.model_path}.tmp"
            with open(tmp_path, 'wb') as f:
                pickle.dump(self.model, f)
            os.replace(tmp_path, self.model_path)
            logger.info(f"✓ Saved model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def predict(self, features: TradeFeatures) -> PredictionResult:
        """Predict outcome for trade with given features."""
        if self.model is None:
            # No model yet, return neutral prediction
            return PredictionResult(
                win_probability=0.5,
                loss_probability=0.5,
                confidence=0.0,
                recommendation='WAIT'
            )

        if not XGBOOST_AVAILABLE:
            return PredictionResult(
                win_probability=0.5,
                loss_probability=0.5,
                confidence=0.0,
                recommendation='WAIT'
            )

        try:
            # Convert features to array
            feature_dict = asdict(features)
            X = np.array([[feature_dict[name] for name in self.feature_names]])

            # Get prediction probabilities
            proba = self.model.predict_proba(X)[0]
            win_prob = float(proba[1]) if len(proba) > 1 else 0.5

            # Make recommendation
            if win_prob > 0.85:
                recommendation = 'ADD_SAFETY'
            elif win_prob > 0.65:
                recommendation = 'WAIT'
            else:
                recommendation = 'BAILOUT'

            return PredictionResult(
                win_probability=win_prob,
                loss_probability=1.0 - win_prob,
                confidence=abs(win_prob - 0.5) * 2,  # High when strongly one way
                recommendation=recommendation
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return PredictionResult(
                win_probability=0.5,
                loss_probability=0.5,
                confidence=0.0,
                recommendation='WAIT'
            )

    def train(self, X: list, y: list):
        """Train model on historical data.

        Args:
            X: List of TradeFeatures objects
            y: List of outcomes (1 for win, 0 for loss)
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not installed: pip install xgboost")
            return

        if len(X) < 20:
            logger.warning(f"Need at least 20 samples to train, got {len(X)}")
            return

        try:
            # Convert to array
            X_array = np.array([
                [getattr(x, name) for name in self.feature_names] for x in X
            ])
            y_array = np.array(y)

            # Train model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            self.model.fit(X_array, y_array)

            # Log results
            train_score = self.model.score(X_array, y_array)
            logger.info(f"✓ Model trained on {len(X)} samples (accuracy: {train_score:.1%})")

            self.save_model()
        except Exception as e:
            logger.error(f"Training error: {e}")


def extract_features_from_trade(trade_data: dict) -> Optional[TradeFeatures]:
    """Extract features from trade data for prediction."""
    try:
        return TradeFeatures(
            volume_ratio=trade_data.get('volume_ratio', 1.0),
            price_change_pct=trade_data.get('price_change_pct', 0.0),
            momentum_score=trade_data.get('momentum_score', 50.0),
            volatility_pct=trade_data.get('volatility_pct', 1.0),
            spread_bps=trade_data.get('spread_bps', 10.0),
            pattern_success_rate=trade_data.get('pattern_success_rate', 50.0),
            dip_depth_pct=trade_data.get('dip_depth_pct', 0.5),
            entry_confidence=trade_data.get('entry_confidence', 50.0),
            leverage=trade_data.get('leverage', 11),
            candles_since_entry=trade_data.get('candles_since_entry', 0),
            current_loss_pct=trade_data.get('current_loss_pct', 0.0)
        )
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None
