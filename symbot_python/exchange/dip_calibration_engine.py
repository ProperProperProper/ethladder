"""DIP Calibration Engine - Learn and Optimize from Real Trades

Tracks every dip decision and outcome, then automatically calibrates:
- Pattern success rates
- Dimension weights
- Decision thresholds
- Confidence intervals
"""

from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import statistics
import pathlib

logger = logging.getLogger(__name__)


@dataclass
class DipTradeRecord:
    """Record of a single dip trade."""
    timestamp: float
    entry_price: float
    dip_depth_percent: float
    decision_confidence: float
    decision_action: str  # ADD_SAFETY, WAIT, BAILOUT
    patterns_matched: list[str]
    dimension_scores: dict[str, float]
    # "walk_forward" (forward_omlx_tester.py's OMLX-consulted simulation)
    # or "live_paper" (dca_bot.py's real/paper engine) — lets the
    # dashboard show which source each decision came from, and lets
    # calibration math be filtered to one source if the two ever need to
    # be compared rather than blended. "unknown" only for records loaded
    # from a state file saved before this field existed.
    source: str = "unknown"

    # Outcome data
    outcome_time: Optional[float] = None
    outcome_bounced: Optional[bool] = None  # Did it reach TP?
    outcome_max_depth: Optional[float] = None  # How deep did it go?
    outcome_recovery_candles: Optional[int] = None  # Time to reach TP
    outcome_safety_orders_needed: Optional[int] = None  # How many safeties used?
    outcome_pnl: Optional[float] = None  # P&L on this trade


class DipCalibrationEngine:
    """Learn from real dips and optimize system."""

    # Hard cap on persisted individual trade records — same reasoning as
    # TradeMemory.MAX_TRADES (see trade_memory.py): unbounded growth here
    # crashed this machine once already via a different file. Recent
    # records are what the dashboard needs; calibration math itself
    # already only cares about the aggregated stats below, not the raw
    # list length.
    MAX_TRADES = 5_000
    # Same reasoning for the discrete training-events timeline.
    MAX_TRAINING_EVENTS = 2_000

    def __init__(self, state_file: str = "dip_calibration_state.json"):
        """Initialize calibration engine."""
        self.state_file = pathlib.Path(state_file)
        self.trades: list[DipTradeRecord] = []
        # Append-only log of calibration events — one entry per
        # record_outcome() call (i.e. per _calibrate_from_outcomes()
        # run), so the dashboard can show WHEN learning happened and
        # what changed, not just the current snapshot. _save_state()
        # only ever persisted the current weight numbers; on every
        # restart every individual decision/outcome and any sense of
        # "when did this last actually learn something" vanished with
        # zero trace beyond those numbers.
        self.training_events: list[dict] = []

        # Pattern calibration
        self.pattern_success_rates: dict[str, float] = {}
        self.pattern_sample_sizes: dict[str, int] = {}
        self.pattern_accuracy: dict[str, float] = {}

        # Dimension weight optimization
        self.dimension_accuracy: dict[str, float] = {}
        self.recommended_weights: dict[str, float] = {}

        # Decision threshold calibration
        self.decision_threshold_stats: dict[str, dict] = {}

        # Load existing state
        self._load_state()

    def record_decision(
        self,
        entry_price: float,
        dip_depth_percent: float,
        decision_confidence: float,
        decision_action: str,
        patterns_matched: list[str],
        dimension_scores: dict[str, float],
        source: str = "unknown",
    ) -> DipTradeRecord:
        """Record a dip decision at entry time."""

        record = DipTradeRecord(
            timestamp=datetime.utcnow().timestamp(),
            entry_price=entry_price,
            dip_depth_percent=dip_depth_percent,
            decision_confidence=decision_confidence,
            decision_action=decision_action,
            patterns_matched=patterns_matched,
            dimension_scores=dimension_scores.copy(),
            source=source,
        )

        self.trades.append(record)
        if len(self.trades) > self.MAX_TRADES:
            self.trades = self.trades[-self.MAX_TRADES:]
        logger.info(
            f"DIP DECISION RECORDED | "
            f"Price: {entry_price:.2f} | "
            f"Dip: {dip_depth_percent:.2f}% | "
            f"Action: {decision_action} | "
            f"Conf: {decision_confidence:.0f}% | "
            f"Patterns: {', '.join(patterns_matched)}"
        )

        return record

    def record_outcome(
        self,
        dip_record: DipTradeRecord,
        bounced: bool,
        max_depth_percent: float,
        recovery_candles: int,
        safety_orders_needed: int,
        pnl: float
    ) -> None:
        """Record outcome of a dip trade."""

        dip_record.outcome_time = datetime.utcnow().timestamp()
        dip_record.outcome_bounced = bounced
        dip_record.outcome_max_depth = max_depth_percent
        dip_record.outcome_recovery_candles = recovery_candles
        dip_record.outcome_safety_orders_needed = safety_orders_needed
        dip_record.outcome_pnl = pnl

        trade_duration = (
            dip_record.outcome_time - dip_record.timestamp
        ) / 60  # minutes

        action = "✅ SUCCESS" if bounced else "❌ FAILED"
        logger.info(
            f"DIP OUTCOME {action} | "
            f"Conf: {dip_record.decision_confidence:.0f}% | "
            f"Action: {dip_record.decision_action} | "
            f"Depth: {max_depth_percent:.2f}% | "
            f"Time: {trade_duration:.0f}m | "
            f"PnL: {pnl:.2f}%"
        )

        # Trigger calibration
        self._calibrate_from_outcomes()
        self._save_state()

    def _calibrate_from_outcomes(self) -> None:
        """Recalibrate system from all recorded outcomes."""

        completed_trades = [t for t in self.trades if t.outcome_bounced is not None]

        if not completed_trades:
            return

        logger.info(f"Calibrating from {len(completed_trades)} completed trades...")

        before_weights = dict(self.recommended_weights)

        # 1. Pattern calibration
        self._calibrate_patterns(completed_trades)

        # 2. Dimension calibration
        self._calibrate_dimensions(completed_trades)

        # 3. Decision threshold calibration
        self._calibrate_thresholds(completed_trades)

        self.get_recommended_weights()  # refresh self.recommended_weights from the calibration just above

        # Record this as a discrete, timestamped event — not just an
        # updated snapshot — so "is this actually learning on a
        # schedule" has real history to show, not just a current number.
        overall_bounce_rate = sum(1 for t in completed_trades if t.outcome_bounced) / len(completed_trades)
        weight_delta = sum(
            abs(self.recommended_weights.get(k, 0) - before_weights.get(k, 0))
            for k in set(self.recommended_weights) | set(before_weights)
        )
        by_source: dict[str, int] = {}
        for t in completed_trades:
            by_source[t.source] = by_source.get(t.source, 0) + 1
        self.training_events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "completed_trades": len(completed_trades),
            "overall_bounce_rate": overall_bounce_rate,
            "patterns_tracked": len(self.pattern_success_rates),
            "weight_delta": weight_delta,
            "trades_by_source": by_source,
        })
        if len(self.training_events) > self.MAX_TRAINING_EVENTS:
            self.training_events = self.training_events[-self.MAX_TRAINING_EVENTS:]

        # Log recommendations
        self._log_recommendations()

    def _calibrate_patterns(self, trades: list[DipTradeRecord]) -> None:
        """Update pattern success rates from actual outcomes."""

        pattern_outcomes: dict[str, list[bool]] = {}

        for trade in trades:
            for pattern in trade.patterns_matched:
                if pattern not in pattern_outcomes:
                    pattern_outcomes[pattern] = []
                pattern_outcomes[pattern].append(trade.outcome_bounced)

        # Calculate success rates
        for pattern, outcomes in pattern_outcomes.items():
            success_count = sum(outcomes)
            total_count = len(outcomes)
            success_rate = success_count / total_count if total_count > 0 else 0

            self.pattern_success_rates[pattern] = success_rate
            self.pattern_sample_sizes[pattern] = total_count

            logger.debug(
                f"Pattern '{pattern}': {success_rate:.1%} "
                f"({success_count}/{total_count} trades)"
            )

    def _calibrate_dimensions(self, trades: list[DipTradeRecord]) -> None:
        """Measure which dimensions actually predict bounces."""

        if not trades:
            return

        # For each dimension, measure correlation with outcome
        all_dimensions = set()
        for trade in trades:
            all_dimensions.update(trade.dimension_scores.keys())

        for dimension in all_dimensions:
            # Get scores for successful vs failed trades
            success_scores = []
            failure_scores = []

            for trade in trades:
                score = trade.dimension_scores.get(dimension, 0)
                if trade.outcome_bounced:
                    success_scores.append(score)
                else:
                    failure_scores.append(score)

            if not success_scores or not failure_scores:
                continue

            # Calculate separability: how well does this dimension separate outcomes?
            avg_success = statistics.mean(success_scores)
            avg_failure = statistics.mean(failure_scores)

            if avg_failure == 0:
                accuracy = 100
            else:
                # How much better are successful trades on this dimension?
                accuracy = min(100, (avg_success / avg_failure) * 100)

            self.dimension_accuracy[dimension] = accuracy

            logger.debug(
                f"Dimension '{dimension}': {accuracy:.0f}% accuracy "
                f"(success avg: {avg_success:.0f}, failure avg: {avg_failure:.0f})"
            )

    def _calibrate_thresholds(self, trades: list[DipTradeRecord]) -> None:
        """Find optimal decision thresholds."""

        for action in ["ADD_SAFETY_AGGRESSIVE", "ADD_SAFETY", "WAIT_1_CANDLE", "BAILOUT"]:
            action_trades = [t for t in trades if t.decision_action == action]

            if not action_trades:
                continue

            outcomes = [t.outcome_bounced for t in action_trades]
            success_rate = sum(outcomes) / len(outcomes) if outcomes else 0
            success_count = sum(outcomes)

            self.decision_threshold_stats[action] = {
                "success_rate": success_rate,
                "success_count": success_count,
                "total_count": len(action_trades),
                "avg_confidence": statistics.mean(
                    [t.decision_confidence for t in action_trades]
                )
            }

    def _log_recommendations(self) -> None:
        """Log calibration recommendations."""

        logger.info("=" * 80)
        logger.info("DIP CALIBRATION REPORT")
        logger.info("=" * 80)

        # Pattern recommendations
        logger.info("\n📊 PATTERN SUCCESS RATES:")
        for pattern in sorted(self.pattern_success_rates.keys()):
            rate = self.pattern_success_rates[pattern]
            count = self.pattern_sample_sizes[pattern]
            logger.info(f"  {pattern}: {rate:.1%} ({count} samples)")

        # Dimension recommendations
        logger.info("\n📈 DIMENSION PREDICTIVENESS (higher = better):")
        sorted_dims = sorted(
            self.dimension_accuracy.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for dim, accuracy in sorted_dims:
            logger.info(f"  {dim}: {accuracy:.0f}%")

        # Decision threshold analysis
        logger.info("\n🎯 DECISION OUTCOME ANALYSIS:")
        for action in ["ADD_SAFETY_AGGRESSIVE", "ADD_SAFETY", "WAIT_1_CANDLE", "BAILOUT"]:
            if action in self.decision_threshold_stats:
                stats = self.decision_threshold_stats[action]
                logger.info(
                    f"  {action}: {stats['success_rate']:.1%} "
                    f"({stats['success_count']}/{stats['total_count']}) "
                    f"avg confidence {stats['avg_confidence']:.0f}%"
                )

        # Overall statistics
        logger.info("\n📋 OVERALL STATISTICS:")
        logger.info(f"  Total dips analyzed: {len(self.trades)}")
        completed = [t for t in self.trades if t.outcome_bounced is not None]
        if completed:
            overall_success = sum(1 for t in completed if t.outcome_bounced)
            logger.info(f"  Overall bounce rate: {overall_success}/{len(completed)} = "
                       f"{overall_success/len(completed):.1%}")

    def get_recommended_weights(self) -> dict[str, float]:
        """Get recommended dimension weights based on calibration."""

        if not self.dimension_accuracy:
            # Return defaults if not calibrated yet
            return {
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

        # Normalize accuracy scores to weights
        total_accuracy = sum(self.dimension_accuracy.values())
        recommended = {}

        for dim, accuracy in self.dimension_accuracy.items():
            # Weight by relative accuracy, but keep some baseline
            weight = (accuracy / total_accuracy) * 0.8 + 0.02
            recommended[dim] = weight

        # Normalize to sum to 1.0
        total = sum(recommended.values())
        for dim in recommended:
            recommended[dim] /= total

        self.recommended_weights = recommended
        return recommended

    def get_status_summary(self) -> dict:
        """Compact status for the dashboard's OMLX Learning tab — the
        headline "is this actually learning" facts, not the full
        calibration internals export_calibration_data() produces."""
        completed = [t for t in self.trades if t.outcome_bounced is not None]
        by_source: dict[str, int] = {}
        for t in completed:
            by_source[t.source] = by_source.get(t.source, 0) + 1
        last_event = self.training_events[-1] if self.training_events else None
        return {
            "total_decisions": len(self.trades),
            "completed_decisions": len(completed),
            "completed_by_source": by_source,
            "overall_bounce_rate": (
                sum(1 for t in completed if t.outcome_bounced) / len(completed)
                if completed else None
            ),
            "patterns_tracked": len(self.pattern_success_rates),
            "training_event_count": len(self.training_events),
            "last_calibrated_at": last_event["timestamp"] if last_event else None,
            "recent_events": self.training_events[-20:],
            "recent_decisions": [asdict(t) for t in self.trades[-20:]],
        }

    def export_calibration_data(self, filepath: str = "dip_calibration_export.json") -> None:
        """Export calibration data for analysis."""

        export = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_trades": len(self.trades),
            "completed_trades": len([t for t in self.trades if t.outcome_bounced is not None]),
            "pattern_success_rates": self.pattern_success_rates,
            "pattern_sample_sizes": self.pattern_sample_sizes,
            "dimension_accuracy": self.dimension_accuracy,
            "recommended_weights": self.recommended_weights,
            "decision_threshold_stats": self.decision_threshold_stats,
            "trade_records": [asdict(t) for t in self.trades]
        }

        # Atomic write (temp file + os.replace) — same reasoning as
        # _save_state() below and every other JSON export in this
        # codebase.
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(export, f, indent=2, default=str)
        os.replace(tmp_path, filepath)

        logger.info(f"Calibration data exported to {filepath}")

    def _save_state(self) -> None:
        """Save calibration state to disk.

        Writes to a temp file then os.replace()s it into place — same
        atomic-write reasoning as TradeMemory.save() (see trade_memory.py):
        this file can now be large enough (up to MAX_TRADES individual
        records) that a plain write is not instantaneous, and a
        concurrent _load_state() elsewhere could otherwise observe a
        truncated/partial file.
        """

        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern_success_rates": self.pattern_success_rates,
            "pattern_sample_sizes": self.pattern_sample_sizes,
            "dimension_accuracy": self.dimension_accuracy,
            "recommended_weights": self.recommended_weights,
            "decision_threshold_stats": self.decision_threshold_stats,
            # Individual records + discrete event history — previously
            # only the snapshot fields above were saved, so a restart
            # lost every trade record and any sense of "when did
            # calibration last actually run" beyond these numbers.
            "trades": [asdict(t) for t in self.trades],
            "training_events": self.training_events,
        }

        try:
            tmp_path = self.state_file.with_suffix(".json.tmp")
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            tmp_path.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save calibration state: {e}")

    def _load_state(self) -> None:
        """Load calibration state from disk."""

        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.pattern_success_rates = state.get("pattern_success_rates", {})
            self.pattern_sample_sizes = state.get("pattern_sample_sizes", {})
            self.dimension_accuracy = state.get("dimension_accuracy", {})
            self.recommended_weights = state.get("recommended_weights", {})
            self.decision_threshold_stats = state.get("decision_threshold_stats", {})
            self.trades = [DipTradeRecord(**t) for t in state.get("trades", [])]
            self.training_events = state.get("training_events", [])

            logger.info(f"Loaded calibration state from {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to load calibration state: {e}")


# Global calibration engine
_calibration_engine: Optional[DipCalibrationEngine] = None


async def get_calibration_engine() -> DipCalibrationEngine:
    """Get or create global calibration engine."""
    global _calibration_engine
    if _calibration_engine is None:
        _calibration_engine = DipCalibrationEngine()
    return _calibration_engine
