"""Bootstrap Trainer - Create initial ML models with synthetic data.

Generates synthetic trade data based on domain knowledge to bootstrap models.
Once real trades are collected, models can be retrained.
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple
import random

from symbot_python.ml.outcome_predictor import OutcomePredictor, TradeFeatures
from symbot_python.ml.rl_decision_optimizer import (
    RLDecisionOptimizer, state_from_market_data, calculate_reward
)

logger = logging.getLogger(__name__)


def generate_synthetic_trades(num_samples: int = 200) -> List[dict]:
    """Generate synthetic trade data based on domain knowledge.

    Creates realistic trade scenarios for ETH ladder with 11x leverage.
    """
    trades = []

    for i in range(num_samples):
        # Realistic scenarios for ETH dip trading
        scenarios = [
            # Morning dip - high success
            {
                'name': 'morning_dip',
                'volatility': random.uniform(0.3, 1.5),
                'momentum': random.uniform(20, 45),
                'pattern_rate': 92,
                'win_rate': 0.92,
            },
            # Support bounce - high success
            {
                'name': 'support_bounce',
                'volatility': random.uniform(0.5, 2.0),
                'momentum': random.uniform(30, 50),
                'pattern_rate': 94,
                'win_rate': 0.94,
            },
            # RSI oversold - medium success
            {
                'name': 'rsi_oversold',
                'volatility': random.uniform(1.0, 3.0),
                'momentum': random.uniform(15, 35),
                'pattern_rate': 89,
                'win_rate': 0.89,
            },
            # Volatile - lower success
            {
                'name': 'volatility_extreme',
                'volatility': random.uniform(2.5, 4.5),
                'momentum': random.uniform(40, 60),
                'pattern_rate': 87,
                'win_rate': 0.87,
            },
            # Bad conditions
            {
                'name': 'liquidation_cascade',
                'volatility': random.uniform(3.0, 5.0),
                'momentum': random.uniform(55, 75),
                'pattern_rate': 45,
                'win_rate': 0.45,
            },
        ]

        scenario = random.choice(scenarios)

        # Determine if trade wins/loses based on scenario win rate
        pnl = random.random() < scenario['win_rate']

        # Generate trade
        trade = {
            'scenario': scenario['name'],
            'volume_ratio': random.uniform(0.8, 2.5),
            'price_change_pct': -random.uniform(0.2, 3.0),  # Negative = dip
            'momentum_score': scenario['momentum'],
            'volatility_pct': scenario['volatility'],
            'spread_bps': random.uniform(5, 20),
            'pattern_success_rate': scenario['pattern_rate'],
            'dip_depth_pct': random.uniform(0.3, 2.0),
            'entry_confidence': random.uniform(60, 95) if pnl else random.uniform(40, 75),
            'leverage': 11,
            'candles_since_entry': random.randint(0, 20),
            'current_loss_pct': random.uniform(0, 2.5) if pnl else random.uniform(0.5, 3.0),
            'pnl_pct': random.uniform(2, 8) if pnl else -random.uniform(1, 4),
            'was_win': pnl,
            'action_taken': random.choice(['ADD_SAFETY', 'WAIT']) if pnl else random.choice(['WAIT', 'BAILOUT']),
            'market_data_at_entry': {
                'volatility_pct': scenario['volatility'],
                'momentum_score': scenario['momentum'],
                'pattern_success_rate': scenario['pattern_rate'],
                'current_loss_pct': 0,
                'candles_since_entry': 0,
            },
            'market_data_at_exit': {
                'volatility_pct': scenario['volatility'] + random.uniform(-0.5, 0.5),
                'momentum_score': scenario['momentum'] + random.uniform(-10, 10),
                'pattern_success_rate': scenario['pattern_rate'],
                'current_loss_pct': random.uniform(0, 2.5),
                'candles_since_entry': random.randint(1, 20),
            },
            'avoided_loss_pct': random.uniform(0, 5) if not pnl else 0,
        }
        trades.append(trade)

    logger.info(f"Generated {num_samples} synthetic trades")
    return trades


def bootstrap_models():
    """Bootstrap ML models with synthetic data."""
    logger.info("Bootstrapping ML models with synthetic training data...")

    # Generate synthetic trades
    trades = generate_synthetic_trades(200)

    # Prepare training data for outcome predictor
    X, y = [], []
    for trade in trades:
        features = TradeFeatures(
            volume_ratio=trade['volume_ratio'],
            price_change_pct=trade['price_change_pct'],
            momentum_score=trade['momentum_score'],
            volatility_pct=trade['volatility_pct'],
            spread_bps=trade['spread_bps'],
            pattern_success_rate=trade['pattern_success_rate'],
            dip_depth_pct=trade['dip_depth_pct'],
            entry_confidence=trade['entry_confidence'],
            leverage=trade['leverage'],
            candles_since_entry=trade['candles_since_entry'],
            current_loss_pct=trade['current_loss_pct']
        )
        X.append(features)
        y.append(1 if trade['was_win'] else 0)

    # Train outcome predictor
    logger.info("Training outcome predictor...")
    predictor = OutcomePredictor()
    predictor.train(X, y)

    # Train RL optimizer
    logger.info("Training RL decision optimizer...")
    rl_optimizer = RLDecisionOptimizer()

    for trade in trades:
        entry_state = state_from_market_data(trade['market_data_at_entry'])
        exit_state = state_from_market_data(trade['market_data_at_exit'])
        action = trade['action_taken']

        outcome = {
            'pnl_pct': trade['pnl_pct'],
            'avoided_loss_pct': trade['avoided_loss_pct']
        }
        reward = calculate_reward(action, outcome)

        rl_optimizer.learn(entry_state, action, reward, exit_state)

    rl_optimizer.save_model()

    logger.info(f"✓ Bootstrap complete")
    logger.info(f"  - Outcome predictor trained on {len(X)} samples")
    logger.info(f"  - RL optimizer learned {len(rl_optimizer.Q_table)} states")

    return predictor, rl_optimizer


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    bootstrap_models()
