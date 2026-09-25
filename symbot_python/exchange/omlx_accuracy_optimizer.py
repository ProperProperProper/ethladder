"""OMLX Accuracy Optimizer - Focus on win rate and loss minimization

Metrics tracked:
- Accuracy (TP/(TP+FP))
- Win rate (wins/total)
- Loss rate (losses/total)
- False positive rate (wrong predictions)
- Drawdown recovery success
- Average loss size
- Loss prevention triggers

Strategies:
1. High confidence only (>80%)
2. Early loss detection (bailout at -2% before -3%)
3. Pattern matching (avoid failed patterns)
4. Loss minimization (exit losing trades earlier)
5. Accuracy feedback loop
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy and loss tracking."""
    true_positives: int = 0  # Correct ADD_SAFETY decisions that won
    false_positives: int = 0  # ADD_SAFETY that lost
    true_negatives: int = 0  # Correct WAIT/BAILOUT decisions
    false_negatives: int = 0  # Missed profitable opportunities

    total_wins: int = 0
    total_losses: int = 0
    total_trades: int = 0

    # Loss tracking
    win_pnl_list: list = field(default_factory=list)
    loss_pnl_list: list = field(default_factory=list)

    # Drawdown specific
    dd_recovery_attempts: int = 0
    dd_recovery_success: int = 0
    dd_recovery_failed: int = 0

    # Confidence correlation
    high_conf_accuracy: float = 0.0  # >80% confidence
    med_conf_accuracy: float = 0.0   # 50-80%
    low_conf_accuracy: float = 0.0   # <50%

    def calculate_accuracy(self) -> float:
        """True positives / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def calculate_precision(self) -> float:
        """TP / (TP + FP) - when OMLX says BUY, how often is it right?"""
        return self.calculate_accuracy()

    def calculate_recall(self) -> float:
        """TP / (TP + FN) - of all profitable opportunities, how many did we catch?"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def calculate_f1_score(self) -> float:
        """Harmonic mean of precision and recall"""
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_avg_win(self) -> float:
        """Average P&L on winning trades"""
        return statistics.mean(self.win_pnl_list) if self.win_pnl_list else 0.0

    def calculate_avg_loss(self) -> float:
        """Average P&L on losing trades (negative)"""
        return statistics.mean(self.loss_pnl_list) if self.loss_pnl_list else 0.0

    def calculate_win_loss_ratio(self) -> float:
        """Avg win / Avg loss magnitude"""
        avg_loss = abs(self.calculate_avg_loss())
        if avg_loss == 0:
            return float('inf')
        return self.calculate_avg_win() / avg_loss

    def get_report(self) -> dict:
        """Generate accuracy report"""
        return {
            "accuracy": self.calculate_accuracy(),
            "precision": self.calculate_precision(),
            "recall": self.calculate_recall(),
            "f1_score": self.calculate_f1_score(),
            "win_rate": self.total_wins / self.total_trades if self.total_trades else 0,
            "loss_rate": self.total_losses / self.total_trades if self.total_trades else 0,
            "avg_win": self.calculate_avg_win(),
            "avg_loss": self.calculate_avg_loss(),
            "win_loss_ratio": self.calculate_win_loss_ratio(),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
            "dd_recovery_success_rate": (
                self.dd_recovery_success / self.dd_recovery_attempts
                if self.dd_recovery_attempts else 0
            ),
            "high_conf_accuracy": self.high_conf_accuracy,
            "med_conf_accuracy": self.med_conf_accuracy,
            "low_conf_accuracy": self.low_conf_accuracy,
        }


class OMLXAccuracyOptimizer:
    """Optimize OMLX for accuracy and loss minimization."""

    def __init__(self):
        """Initialize optimizer."""
        self.metrics = AccuracyMetrics()
        self.high_conf_trades = []  # >80% confidence
        self.med_conf_trades = []   # 50-80%
        self.low_conf_trades = []   # <50%
        self.failed_patterns = {}   # Track which patterns fail
        self.loss_prevention_rules = []

    def record_trade_decision(
        self,
        decision_id: str,
        confidence: float,
        recommendation: str,
        patterns_matched: list[str],
        entry_price: float,
    ) -> None:
        """Record a trading decision."""

        trade_info = {
            "id": decision_id,
            "confidence": confidence,
            "recommendation": recommendation,
            "patterns": patterns_matched,
            "entry_price": entry_price,
            "timestamp": datetime.now().isoformat(),
            "outcome": None,  # To be filled later
            "pnl": None,
        }

        # Categorize by confidence
        if confidence > 80:
            self.high_conf_trades.append(trade_info)
        elif confidence >= 50:
            self.med_conf_trades.append(trade_info)
        else:
            self.low_conf_trades.append(trade_info)

    def record_trade_outcome(
        self,
        decision_id: str,
        was_profitable: bool,
        pnl_percent: float,
        actual_bounce: bool,
    ) -> None:
        """Record outcome of a decision."""

        # Find and update the trade record
        for trades_list in [self.high_conf_trades, self.med_conf_trades, self.low_conf_trades]:
            for trade in trades_list:
                if trade["id"] == decision_id:
                    trade["outcome"] = "WIN" if was_profitable else "LOSS"
                    trade["pnl"] = pnl_percent
                    trade["actual_bounce"] = actual_bounce

                    # Update metrics
                    self._update_metrics(trade, was_profitable, actual_bounce)
                    return

    def _update_metrics(self, trade: dict, was_profitable: bool, actual_bounce: bool) -> None:
        """Update accuracy metrics."""

        # Track wins/losses
        self.metrics.total_trades += 1
        if was_profitable:
            self.metrics.total_wins += 1
            self.metrics.win_pnl_list.append(trade["pnl"])
        else:
            self.metrics.total_losses += 1
            self.metrics.loss_pnl_list.append(trade["pnl"])

        # Track TP/FP/TN/FN. ADD_SAFETY_AGGRESSIVE (fires only above the
        # ~85% confidence band — see BounceAnalyzer's recommendation
        # thresholds) is just as much a "buy" decision as plain
        # ADD_SAFETY for this purpose; excluding it would silently
        # undercount true/false positives for the highest-confidence
        # decisions specifically, the ones this accuracy tracking exists
        # to validate most.
        is_add_safety = trade["recommendation"] in ("ADD_SAFETY", "ADD_SAFETY_AGGRESSIVE")

        if is_add_safety:
            if was_profitable:
                self.metrics.true_positives += 1
            else:
                self.metrics.false_positives += 1
        else:
            if not was_profitable:
                self.metrics.true_negatives += 1
            else:
                self.metrics.false_negatives += 1

        # Track confidence correlation
        if trade["confidence"] > 80:
            self.metrics.high_conf_accuracy = (
                self.metrics.true_positives / max(1, self.metrics.true_positives + self.metrics.false_positives)
            )
        elif trade["confidence"] >= 50:
            self.metrics.med_conf_accuracy = (
                self.metrics.true_positives / max(1, self.metrics.true_positives + self.metrics.false_positives)
            )
        else:
            self.metrics.low_conf_accuracy = (
                self.metrics.true_positives / max(1, self.metrics.true_positives + self.metrics.false_positives)
            )

        # Track failed patterns
        for pattern in trade.get("patterns", []):
            if not was_profitable:
                if pattern not in self.failed_patterns:
                    self.failed_patterns[pattern] = {"fails": 0, "attempts": 0}
                self.failed_patterns[pattern]["fails"] += 1
                self.failed_patterns[pattern]["attempts"] += 1
            else:
                if pattern not in self.failed_patterns:
                    self.failed_patterns[pattern] = {"fails": 0, "attempts": 0}
                self.failed_patterns[pattern]["attempts"] += 1

        # Track drawdown recovery
        if trade.get("actual_bounce"):
            self.metrics.dd_recovery_attempts += 1
            if was_profitable:
                self.metrics.dd_recovery_success += 1
            else:
                self.metrics.dd_recovery_failed += 1

    def get_accuracy_report(self) -> dict:
        """Get detailed accuracy report."""
        report = self.metrics.get_report()

        # Add pattern performance
        pattern_performance = {}
        for pattern, stats in self.failed_patterns.items():
            fail_rate = stats["fails"] / stats["attempts"] if stats["attempts"] else 0
            pattern_performance[pattern] = {
                "attempts": stats["attempts"],
                "failures": stats["fails"],
                "fail_rate": fail_rate,
                "success_rate": 1 - fail_rate,
                "status": "⚠️ AVOID" if fail_rate > 0.3 else "✓ OK",
            }

        report["patterns"] = pattern_performance
        report["timestamp"] = datetime.now().isoformat()

        return report

    def get_loss_prevention_rules(self) -> list[str]:
        """Generate loss prevention rules based on data."""
        rules = []

        # Rule 1: Avoid low-confidence trades
        if self.metrics.low_conf_accuracy < 0.5:
            rules.append("RULE 1: Avoid trades <50% confidence (accuracy < 50%)")

        # Rule 2: Failed patterns
        for pattern, stats in self.failed_patterns.items():
            if stats["attempts"] >= 5:  # Only if we have sample size
                fail_rate = stats["fails"] / stats["attempts"]
                if fail_rate > 0.4:
                    rules.append(f"RULE 2: Avoid '{pattern}' pattern ({fail_rate:.0%} failure rate)")

        # Rule 3: Loss size
        avg_loss = abs(self.metrics.calculate_avg_loss())
        avg_win = self.metrics.calculate_avg_win()
        if avg_loss > avg_win:
            rules.append(f"RULE 3: Average loss (${avg_loss:.2f}%) > average win (${avg_win:.2f}%)")
            rules.append("        → Need to exit losing trades earlier (at -1% instead of -1.3%)")

        # Rule 4: Drawdown recovery
        if self.metrics.dd_recovery_attempts >= 10:
            recovery_rate = self.metrics.dd_recovery_success / self.metrics.dd_recovery_attempts
            if recovery_rate < 0.8:
                rules.append(
                    f"RULE 4: Drawdown recovery success only {recovery_rate:.0%} "
                    f"({self.metrics.dd_recovery_success}/{self.metrics.dd_recovery_attempts})"
                )
                rules.append("        → Increase safety order aggressiveness")

        # Rule 5: Win rate
        win_rate = self.metrics.total_wins / self.metrics.total_trades if self.metrics.total_trades else 0
        if win_rate < 0.80:
            rules.append(f"RULE 5: Win rate only {win_rate:.0%} - need {0.80:.0%} for 11x leverage")

        return rules

    def should_skip_trade(self, confidence: float, patterns: list[str]) -> tuple[bool, str]:
        """Determine if we should skip this trade for accuracy.

        Returns:
            (skip: bool, reason: str)
        """

        # Skip if confidence too low
        if confidence < 50:
            return True, f"Confidence {confidence:.0f}% < 50% threshold"

        # Skip if patterns have high failure rate
        for pattern in patterns:
            if pattern in self.failed_patterns:
                stats = self.failed_patterns[pattern]
                if stats["attempts"] >= 5:
                    fail_rate = stats["fails"] / stats["attempts"]
                    if fail_rate > 0.4:
                        return True, f"Pattern '{pattern}' has {fail_rate:.0%} failure rate"

        return False, ""

    def should_exit_early(self, current_loss_percent: float, avg_historical_loss: float) -> bool:
        """Should we exit losing trade before max drawdown?

        Early exit prevents losses from spiraling.
        """
        # If current loss is 75% of historical average loss, exit early
        if current_loss_percent < 0:
            loss_magnitude = abs(current_loss_percent)
            historical_loss = abs(avg_historical_loss)

            if historical_loss > 0 and loss_magnitude >= historical_loss * 0.75:
                return True

        return False

    def recommend_confidence_threshold(self) -> float:
        """Recommend optimal confidence threshold for trading.

        Returns:
            Confidence threshold (e.g., 0.75 = 75%)
        """

        # Find confidence level with best accuracy
        accuracies = {
            "high": self.metrics.high_conf_accuracy,
            "med": self.metrics.med_conf_accuracy,
            "low": self.metrics.low_conf_accuracy,
        }

        if self.metrics.high_conf_accuracy > 0.85:
            return 0.80  # High confidence trades are good
        elif self.metrics.med_conf_accuracy > 0.75:
            return 0.50  # Medium confidence trades are OK
        else:
            return 0.90  # Only trade highest confidence

    def export_accuracy_report(self, filename: str = None) -> str:
        """Export detailed accuracy report with full trade data for ML learning."""
        if not filename:
            filename = f"accuracy_report_{int(datetime.now().timestamp())}.json"

        import json

        report = self.get_accuracy_report()
        loss_prevention = self.get_loss_prevention_rules()
        recommended_threshold = self.recommend_confidence_threshold()

        # Combine all trades with confidence levels
        all_trades = []
        for trade in self.high_conf_trades:
            trade['confidence_level'] = 'high'
            all_trades.append(trade)
        for trade in self.med_conf_trades:
            trade['confidence_level'] = 'medium'
            all_trades.append(trade)
        for trade in self.low_conf_trades:
            trade['confidence_level'] = 'low'
            all_trades.append(trade)

        full_report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": report,
            "loss_prevention_rules": loss_prevention,
            "recommended_confidence_threshold": recommended_threshold,
            "total_trades_analyzed": self.metrics.total_trades,
            "trade_records": {
                "high_confidence": len(self.high_conf_trades),
                "medium_confidence": len(self.med_conf_trades),
                "low_confidence": len(self.low_conf_trades),
            },
            "trades": all_trades,  # Full trade data for ML training
        }

        # Atomic write (temp file + os.replace) — same reasoning as
        # every other JSON export in this codebase: this file is read
        # every 30s by run_everything.py's omlx_metrics_exporter_loop
        # (always the newest forward_test_accuracy_*.json), and can be
        # tens of MB. A plain in-place write left a real window for that
        # reader to observe a partial file mid-write — confirmed
        # directly: JSONDecodeError ("Expecting property name enclosed
        # in double quotes") at line 1,883,242 of one of these files.
        import os as _os
        tmp_path = f"{filename}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)
        _os.replace(tmp_path, filename)

        logger.info(f"Accuracy report exported to {filename} ({len(all_trades)} trades for ML learning)")
        return filename

    def print_summary(self) -> None:
        """Print accuracy summary to console."""
        report = self.get_accuracy_report()

        logger.info("=" * 80)
        logger.info("OMLX ACCURACY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Trades: {self.metrics.total_trades}")
        logger.info(f"Win Rate: {report['win_rate']:.1%}")
        logger.info(f"Accuracy: {report['accuracy']:.1%}")
        logger.info(f"Precision: {report['precision']:.1%}")
        logger.info(f"Recall: {report['recall']:.1%}")
        logger.info(f"F1 Score: {report['f1_score']:.2f}")
        logger.info("")
        logger.info(f"Avg Win: {report['avg_win']:+.2f}%")
        logger.info(f"Avg Loss: {report['avg_loss']:+.2f}%")
        logger.info(f"Win/Loss Ratio: {report['win_loss_ratio']:.2f}x")
        logger.info("")
        logger.info(f"Confidence Accuracy:")
        logger.info(f"  >80%: {report['high_conf_accuracy']:.1%}")
        logger.info(f"  50-80%: {report['med_conf_accuracy']:.1%}")
        logger.info(f"  <50%: {report['low_conf_accuracy']:.1%}")
        logger.info("")

        rules = self.get_loss_prevention_rules()
        if rules:
            logger.info("Loss Prevention Rules:")
            for rule in rules:
                logger.info(f"  {rule}")
        logger.info("=" * 80)


async def get_accuracy_optimizer() -> OMLXAccuracyOptimizer:
    """Get or create optimizer instance."""
    return OMLXAccuracyOptimizer()
