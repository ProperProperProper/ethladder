"""Training Pipeline - Collect data and train ML models."""

import json
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from symbot_python.ml.outcome_predictor import OutcomePredictor, TradeFeatures
from symbot_python.ml.rl_decision_optimizer import (
    RLDecisionOptimizer, state_from_market_data, calculate_reward
)

logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Orchestrates data collection and model training."""

    def __init__(self):
        """Initialize training pipeline."""
        self.outcome_predictor = OutcomePredictor()
        self.rl_optimizer = RLDecisionOptimizer()

        self.collected_trades = []
        self.training_log = []

    def load_forward_test_data(self, accuracy_file: str) -> bool:
        """Load trade data from forward test accuracy report."""
        try:
            with open(accuracy_file, 'r') as f:
                data = json.load(f)

            trades = data.get('trades', [])
            if not trades:
                logger.warning("No trades found in report")
                return False

            logger.info(f"Loaded {len(trades)} trades from {accuracy_file}")

            # Convert to training format
            for trade in trades:
                self.collected_trades.append(trade)

            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def prepare_outcome_training_data(self) -> Tuple[List[TradeFeatures], List[int]]:
        """Prepare data for outcome predictor training."""
        X, y = [], []

        for trade in self.collected_trades:
            try:
                # Extract features
                features = TradeFeatures(
                    volume_ratio=trade.get('volume_ratio', 1.0),
                    price_change_pct=trade.get('price_change_pct', 0.0),
                    momentum_score=trade.get('momentum_score', 50.0),
                    volatility_pct=trade.get('volatility_pct', 1.0),
                    spread_bps=trade.get('spread_bps', 10.0),
                    pattern_success_rate=trade.get('pattern_success_rate', 50.0),
                    dip_depth_pct=trade.get('dip_depth_pct', 0.5),
                    entry_confidence=trade.get('entry_confidence', 50.0),
                    leverage=trade.get('leverage', 11),
                    candles_since_entry=trade.get('candles_since_entry', 0),
                    current_loss_pct=trade.get('current_loss_pct', 0.0)
                )

                # Label: 1 for win, 0 for loss
                label = 1 if trade.get('pnl_pct', 0) > 0 else 0

                X.append(features)
                y.append(label)
            except Exception as e:
                logger.warning(f"Skipped trade: {e}")

        logger.info(f"Prepared {len(X)} samples for outcome training")
        return X, y

    def train_outcome_predictor(self) -> bool:
        """Train XGBoost outcome predictor."""
        logger.info("Training Outcome Predictor...")

        X, y = self.prepare_outcome_training_data()

        if len(X) < 20:
            logger.warning(f"Not enough data: {len(X)} samples (need 20+)")
            return False

        self.outcome_predictor.train(X, y)

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': 'outcome_predictor',
            'samples': len(X),
            'wins': sum(y),
            'losses': len(y) - sum(y)
        }
        self.training_log.append(log_entry)
        logger.info(f"✓ Trained on {len(X)} samples ({sum(y)} wins, {len(y)-sum(y)} losses)")
        return True

    def train_rl_optimizer(self) -> bool:
        """Train RL decision optimizer from trades."""
        logger.info("Training RL Decision Optimizer...")

        if len(self.collected_trades) < 10:
            logger.warning(f"Not enough trades: {len(self.collected_trades)} (need 10+)")
            return False

        trained_count = 0

        for trade in self.collected_trades:
            try:
                # Get market state at entry
                market_data = trade.get('market_data_at_entry', {})
                entry_state = state_from_market_data(market_data)

                # Get action taken
                action = trade.get('action_taken', 'WAIT')

                # Get market state at exit
                market_data_exit = trade.get('market_data_at_exit', {})
                exit_state = state_from_market_data(market_data_exit)

                # Calculate reward
                outcome = {
                    'pnl_pct': trade.get('pnl_pct', 0),
                    'avoided_loss_pct': trade.get('avoided_loss_pct', 0)
                }
                reward = calculate_reward(action, outcome)

                # Update Q-values
                self.rl_optimizer.learn(entry_state, action, reward, exit_state)
                trained_count += 1
            except Exception as e:
                logger.warning(f"Skipped trade in RL training: {e}")

        self.rl_optimizer.save_model()

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': 'rl_optimizer',
            'trades_learned': trained_count,
            'total_states': len(self.rl_optimizer.Q_table),
            'total_reward': self.rl_optimizer.total_reward
        }
        self.training_log.append(log_entry)
        logger.info(f"✓ Trained on {trained_count} trades ({len(self.rl_optimizer.Q_table)} states)")
        return True

    def train_all_models(self, accuracy_file: str) -> bool:
        """Complete training pipeline."""
        logger.info("Starting ML Training Pipeline")
        logger.info(f"Loading data from: {accuracy_file}")

        # Load data
        if not self.load_forward_test_data(accuracy_file):
            return False

        # Train both models
        success = True
        success = self.train_outcome_predictor() and success
        success = self.train_rl_optimizer() and success

        # Save training log
        self._save_training_log()

        if success:
            logger.info("✓ All models trained successfully")
        else:
            logger.warning("⚠ Some models failed to train")

        return success

    def _save_training_log(self):
        """Save training log."""
        log_file = Path('ml_training_log.json')
        try:
            with open(log_file, 'w') as f:
                json.dump(self.training_log, f, indent=2)
            logger.info(f"✓ Training log saved to {log_file}")
        except Exception as e:
            logger.error(f"Failed to save training log: {e}")

    def get_models(self) -> Tuple[OutcomePredictor, RLDecisionOptimizer]:
        """Get trained models for deployment."""
        return self.outcome_predictor, self.rl_optimizer


def train_from_latest_report():
    """Convenience function to train from latest forward test report."""
    # Find latest accuracy report
    reports = sorted(Path('.').glob('forward_test_accuracy_*.json'), reverse=True)
    if not reports:
        logger.error("No forward test reports found")
        return False

    latest_report = reports[0]
    logger.info(f"Using latest report: {latest_report}")

    pipeline = MLTrainingPipeline()
    return pipeline.train_all_models(str(latest_report))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_from_latest_report()
