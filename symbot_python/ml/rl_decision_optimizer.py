"""Reinforcement Learning Decision Optimizer - Learns optimal trade actions.

Uses Q-Learning to optimize decisions: ADD_SAFETY, WAIT, BAILOUT
Learns from outcomes: profit/loss achieved.
Applicable to ETH ladder strategy with 11x leverage.
"""

import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Discretized market state for Q-learning."""
    volatility_level: int  # 0-3: low, medium, high, extreme
    momentum_level: int  # 0-2: bearish, neutral, bullish
    loss_magnitude: int  # 0-3: none, small (0-1%), medium (1-2%), large (2%+)
    pattern_strength: int  # 0-3: weak, medium, strong, very strong
    time_in_trade: int  # 0-4: 0-5 candles, 5-10, 10-15, 15+ candles, closed

    def to_key(self) -> tuple:
        """Convert to hashable key for Q-table."""
        return (self.volatility_level, self.momentum_level, self.loss_magnitude,
                self.pattern_strength, self.time_in_trade)


class RLDecisionOptimizer:
    """Q-Learning based decision optimizer for trade actions."""

    # Actions available
    ACTIONS = ['ADD_SAFETY', 'WAIT', 'BAILOUT']
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, model_path: str = 'omlx_rl_model.json'):
        """Initialize RL optimizer with Q-table."""
        self.model_path = Path(model_path)

        # Q-table: state -> action -> value
        self.Q_table: Dict[tuple, Dict[str, float]] = {}

        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate

        # Tracking
        self.training_episodes = 0
        self.total_reward = 0.0

        # Load existing model
        if self.model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load Q-table from disk."""
        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                # Convert string keys back to tuples
                self.Q_table = {
                    eval(k): v for k, v in data.get('Q_table', {}).items()
                }
                self.training_episodes = data.get('episodes', 0)
                self.total_reward = data.get('total_reward', 0.0)
            logger.info(f"✓ Loaded RL model with {len(self.Q_table)} states")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            self.Q_table = {}

    def save_model(self):
        """Save Q-table to disk."""
        try:
            data = {
                'Q_table': {str(k): v for k, v in self.Q_table.items()},
                'episodes': self.training_episodes,
                'total_reward': self.total_reward
            }
            # Atomic write (temp file + os.replace) — this file is
            # reloaded live by omlx_ml_advisor.py's RLDecisionOptimizer()
            # (used by real/paper trading decisions) on a different
            # schedule than this one saves on (every ~2 min via
            # ml_trainer_loop) — same race class as the other JSON
            # exports fixed elsewhere in this codebase, where a plain
            # in-place write left a real window for a torn read.
            tmp_path = f"{self.model_path}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.model_path)
            logger.info(f"✓ Saved RL model ({len(self.Q_table)} states)")
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")

    def _get_q_values(self, state: MarketState) -> Dict[str, float]:
        """Get Q-values for all actions in state."""
        state_key = state.to_key()
        if state_key not in self.Q_table:
            # Initialize new state with zero values
            self.Q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        return self.Q_table[state_key]

    def choose_action(self, state: MarketState, training: bool = False) -> str:
        """Choose best action for state (epsilon-greedy)."""
        q_values = self._get_q_values(state)

        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.ACTIONS)
        else:
            # Exploitation: best action
            best_action = max(self.ACTIONS, key=lambda a: q_values[a])
            return best_action

    def learn(self, state: MarketState, action: str, reward: float, next_state: MarketState):
        """Update Q-values based on experience (Q-Learning update rule)."""
        state_key = state.to_key()
        next_state_key = next_state.to_key()

        # Ensure states exist
        if state_key not in self.Q_table:
            self.Q_table[state_key] = {a: 0.0 for a in self.ACTIONS}
        if next_state_key not in self.Q_table:
            self.Q_table[next_state_key] = {a: 0.0 for a in self.ACTIONS}

        # Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        current_q = self.Q_table[state_key][action]
        max_next_q = max(self.Q_table[next_state_key].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        self.Q_table[state_key][action] = new_q
        self.total_reward += reward
        self.training_episodes += 1

    def get_recommendation(self, state: MarketState) -> Tuple[str, float]:
        """Get action recommendation with confidence score."""
        q_values = self._get_q_values(state)
        best_action = max(self.ACTIONS, key=lambda a: q_values[a])
        q_vals_list = [q_values[a] for a in self.ACTIONS]

        # Confidence: how much better is best action vs second best
        q_vals_sorted = sorted(q_vals_list, reverse=True)
        if len(q_vals_sorted) > 1:
            confidence = max(0, q_vals_sorted[0] - q_vals_sorted[1])
        else:
            confidence = 0

        # Normalize confidence to 0-100
        confidence = min(100, max(0, confidence * 10))

        return best_action, confidence


def state_from_market_data(market_data: dict) -> MarketState:
    """Convert market data to discretized state."""
    # Volatility level
    volatility = market_data.get('volatility_pct', 1.0)
    if volatility < 0.5:
        volatility_level = 0
    elif volatility < 1.5:
        volatility_level = 1
    elif volatility < 3.0:
        volatility_level = 2
    else:
        volatility_level = 3

    # Momentum level
    momentum = market_data.get('momentum_score', 50)
    if momentum < 40:
        momentum_level = 0  # Bearish
    elif momentum > 60:
        momentum_level = 2  # Bullish
    else:
        momentum_level = 1  # Neutral

    # Loss magnitude
    loss = market_data.get('current_loss_pct', 0.0)
    if loss < 0.1:
        loss_magnitude = 0
    elif loss < 1.0:
        loss_magnitude = 1
    elif loss < 2.0:
        loss_magnitude = 2
    else:
        loss_magnitude = 3

    # Pattern strength
    pattern_rate = market_data.get('pattern_success_rate', 50)
    if pattern_rate < 60:
        pattern_strength = 0
    elif pattern_rate < 75:
        pattern_strength = 1
    elif pattern_rate < 90:
        pattern_strength = 2
    else:
        pattern_strength = 3

    # Time in trade
    candles = market_data.get('candles_since_entry', 0)
    if candles <= 5:
        time_in_trade = 0
    elif candles <= 10:
        time_in_trade = 1
    elif candles <= 15:
        time_in_trade = 2
    elif candles <= 20:
        time_in_trade = 3
    else:
        time_in_trade = 4

    return MarketState(
        volatility_level=volatility_level,
        momentum_level=momentum_level,
        loss_magnitude=loss_magnitude,
        pattern_strength=pattern_strength,
        time_in_trade=time_in_trade
    )


def calculate_reward(action: str, outcome: dict) -> float:
    """Calculate reward for action taken and outcome achieved.

    Rewards:
    - ADD_SAFETY + WIN: +100
    - ADD_SAFETY + LOSS: -50
    - WAIT + WIN: +50
    - WAIT + LOSS: -50
    - BAILOUT + avoided loss: +30
    - BAILOUT + missed profit: -20
    """
    pnl = outcome.get('pnl_pct', 0.0)
    was_win = pnl > 0

    if action == 'ADD_SAFETY':
        return 100 if was_win else -50
    elif action == 'WAIT':
        return 50 if was_win else -50
    elif action == 'BAILOUT':
        # Bailout is good if loss was avoided, bad if profit missed
        avoided_loss = outcome.get('avoided_loss_pct', 0.0)
        return 30 if avoided_loss > 1.0 else -20
    else:
        return 0
