"""Forward OMLX Tester - Real-time parameter optimization with OMLX learning

Runs 30-minute forward testing loop:
1. Test parameter sets over live/paper data
2. Identify top 5 winning params
3. Re-run top 5 with OMLX consulting EVERY entry/exit
4. OMLX learns from wins/losses
5. Fine-tune OMLX learnings
6. Repeat loop

Separate from bot parameter finder - focuses on OMLX feedback loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional
import statistics
import json

from symbot_python.exchange.dip_calibration_engine import DipCalibrationEngine, get_calibration_engine
from symbot_python.exchange.dip_analysis_service import DipAnalysisService
from symbot_python.exchange.omlx_accuracy_optimizer import OMLXAccuracyOptimizer
from symbot_python.exchange.omlx_ml_advisor import get_ml_advisor
from symbot_python.ml.walk_forward_trainer import continuous_walk_forward_training
from symbot_python.ml.unified_learning_loop import create_unified_loop, TradeEvent
from symbot_python.strategy.market_data_collector import MarketDataCollector, OHLCV
from symbot_python.strategy.backtest import BacktestConfig, run_backtest

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """Trading parameters to test."""
    name: str
    leverage: int
    tp_percent: float  # Take profit %
    safety_trigger_1: float  # First safety trigger %
    safety_size_1: float  # First safety size %
    safety_trigger_2: float  # Second safety trigger %
    safety_size_2: float  # Second safety size %
    base_order_size: float


@dataclass
class ForwardTestResult:
    """Result of one forward test run."""
    param_set: ParameterSet
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_percent: float
    avg_win_pnl: float
    avg_loss_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    omlx_consulted: bool


@dataclass
class OMLXTrade:
    """Trade record with OMLX decision."""
    entry_price: float
    entry_time: float
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl_percent: Optional[float] = None
    omlx_decision: Optional[str] = None  # ADD_SAFETY, WAIT, BAILOUT
    omlx_confidence: Optional[float] = None
    was_profitable: Optional[bool] = None
    ml_enhanced: bool = False  # Was this decision enhanced by ML?
    original_confidence: Optional[float] = None  # Confidence before ML enhancement
    original_decision: Optional[str] = None  # Decision before ML enhancement


class ForwardOMLXTester:
    """Real-time forward tester with OMLX learning loop."""

    def __init__(self, duration_minutes: int = 30, test_interval_seconds: int = 60):
        """Initialize tester.

        Args:
            duration_minutes: How long to run forward tests (default 30)
            test_interval_seconds: Run complete test suite every N seconds
        """
        self.duration_minutes = duration_minutes
        self.test_interval_seconds = test_interval_seconds
        self.start_time = None
        self.end_time = None

        # Testing state
        self.iterations = 0
        self.all_results = []  # All results from all iterations
        self.top_params = []  # Top 5 param sets
        self.omlx_trades = []  # Trades with OMLX feedback

        # OMLX state
        self.calibration = DipCalibrationEngine()
        self.accuracy_optimizer = OMLXAccuracyOptimizer()
        self.dip_service = None
        self.ml_advisor = get_ml_advisor()  # ML enhancement

        # Unified Learning Loop - all components learn from each trade
        self.unified_loop = create_unified_loop(
            omlx=self,
            ml_advisor=self.ml_advisor,
            accuracy_optimizer=self.accuracy_optimizer,
            calibration=self.calibration
        )

        # Metrics tracking
        self.iteration_metrics = []

        # Log ML status
        if self.ml_advisor.models_available:
            logger.info("✓ ML models available - decisions will be enhanced")
        else:
            logger.info("⚠ ML models warming up - will use OMLX only")

        logger.info("✓ Unified Learning Loop initialized - all components active")

    async def run(self, klines_data: list) -> dict:
        """Run 30-minute forward testing loop.

        Args:
            klines_data: Historical kline data (1-minute candles)

        Returns:
            Summary of testing results and OMLX learnings
        """
        self.start_time = time.time()
        self.end_time = self.start_time + (self.duration_minutes * 60)

        logger.info(f"Starting Forward OMLX Tester ({self.duration_minutes} minutes)")
        logger.info(f"Duration: {self.start_time} to {self.end_time}")

        # emit_critical_alerts=False: this replays real historical
        # candles continuously across many parameter combinations —
        # hitting a real >3% historical move is routine backtest
        # activity here, not a live emergency worth a native desktop
        # notification (see DipAnalysisService's own docstring).
        self.dip_service = DipAnalysisService(emit_critical_alerts=False)

        # Use the SAME shared calibration engine dca_bot.py's live/paper
        # engine records to (previously: a throwaway DipCalibrationEngine()
        # constructed in __init__ that nothing ever called record_decision/
        # record_outcome on — every forward_test_calibration_*.json export
        # was always empty defaults). Both sources now genuinely feed one
        # learning loop, distinguished by DipTradeRecord.source. Keep
        # self.unified_loop's reference in sync — it was built in __init__
        # against the old throwaway instance.
        self.calibration = await get_calibration_engine()
        self.unified_loop.calibration = self.calibration

        iteration = 0
        while time.time() < self.end_time:
            iteration += 1
            elapsed = time.time() - self.start_time
            elapsed_min = elapsed / 60

            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} ({elapsed_min:.1f}/{self.duration_minutes} min)")
            logger.info(f"{'='*80}")

            # Generate parameter sets to test
            param_sets = self._generate_param_sets(iteration)

            # Test each parameter set
            iteration_results = []
            for params in param_sets:
                result = await self._test_params(params, klines_data)
                iteration_results.append(result)
                self.all_results.append(result)

            # Identify top 5
            self.top_params = self._get_top_5_params(iteration_results)
            logger.info(f"\nTop 5 params this iteration:")
            for i, (params, result) in enumerate(self.top_params, 1):
                logger.info(
                    f"  {i}. {params.name}: {result.win_rate:.1%} WR, "
                    f"{result.total_pnl_percent:+.2f}% PnL"
                )

            # Re-run top 5 with OMLX consulting
            logger.info(f"\nRe-running top 5 with OMLX consulting...")
            omlx_results = []
            for params, base_result in self.top_params:
                omlx_result = await self._test_params_with_omlx(params, klines_data)
                omlx_results.append((params, omlx_result))

            # Log OMLX improvements
            logger.info(f"\nOMLX Consultation Results:")
            for (params, base), (_, omlx) in zip(self.top_params, omlx_results):
                improvement = omlx.total_pnl_percent - base.total_pnl_percent
                accuracy_improvement = omlx.win_rate - base.win_rate
                logger.info(
                    f"  {params.name}: "
                    f"PnL: {base.total_pnl_percent:+.2f}% → {omlx.total_pnl_percent:+.2f}% ({improvement:+.2f}%) | "
                    f"Accuracy: {base.win_rate:.1%} → {omlx.win_rate:.1%} ({accuracy_improvement:+.1%})"
                )

            # OMLX learns from results
            await self._omlx_learn_from_iteration(iteration_results + [r for _, r in omlx_results])

            # Store iteration metrics
            self.iteration_metrics.append({
                "iteration": iteration,
                "elapsed_min": elapsed_min,
                "params_tested": len(param_sets),
                "best_win_rate": max(r.win_rate for r in iteration_results),
                "best_pnl": max(r.total_pnl_percent for r in iteration_results),
                "omlx_avg_confidence": statistics.mean(
                    [t.omlx_confidence for t in self.omlx_trades if t.omlx_confidence]
                ) if any(t.omlx_confidence for t in self.omlx_trades) else 0
            })

            # Wait before next iteration
            time_to_sleep = self.test_interval_seconds - (time.time() - self.start_time - elapsed)
            if time_to_sleep > 0:
                await asyncio.sleep(time_to_sleep)

        # Generate final report
        return self._generate_report()

    def _generate_param_sets(self, iteration: int) -> list[ParameterSet]:
        """Generate parameter sets to test."""
        param_sets = []

        # Base params (leverage 11x, tight TP)
        leverage_values = [11, 9, 15]
        tp_values = [0.33, 0.25, 0.5]
        safety_trigger_values = [1.3, 0.8, 2.0]
        safety_size_values = [50, 35, 65]

        idx = 0
        for lev in leverage_values:
            for tp in tp_values:
                for st1 in safety_trigger_values:
                    for ss1 in safety_size_values:
                        idx += 1
                        if idx > 12:  # Limit to reasonable number
                            break

                        param_sets.append(ParameterSet(
                            name=f"P{iteration}_{idx}_L{lev}TP{tp}S{st1}",
                            leverage=lev,
                            tp_percent=tp,
                            safety_trigger_1=st1,
                            safety_size_1=ss1,
                            safety_trigger_2=st1 * 2,
                            safety_size_2=ss1 * 0.5,
                            base_order_size=0.1
                        ))

        return param_sets[:12]  # Return max 12 params per iteration

    async def _test_params(
        self,
        params: ParameterSet,
        klines_data: list
    ) -> ForwardTestResult:
        """Test a single parameter set using backtest.py's real engine —
        real stop-loss/liquidation/take-profit modeling and the actual
        DCA safety-order ladder, not a standalone approximation.

        Previously this simulated every candle as its own isolated
        "buy now, sell at next candle's high" trade — exit_price was
        always next_candle.high (the best possible price) against
        entry_price = this_candle.close. Since candles are back-to-back
        with no gaps, next.high >= this.close holds for essentially
        every real candle, so a "loss" could only ever be the exact tie
        (high == close) — never a genuine negative number. Confirmed
        directly against 499 real ETHUSDT candles: 0 strictly negative
        outcomes, only exact-zero ties recorded as losses. Every trade
        the ML models learned from taught them safety orders never
        really cost anything. backtest.py's run_backtest() has none of
        that — real per-bar liquidation/stop-loss checks in a fixed,
        pessimistic evaluation order (see its own module docstring's
        "ANTI-REPAINTING" section) and an actual multi-rung ladder.
        """
        logger.info(f"🔥 _test_params called for {params.name}")

        config = self._backtest_config_from_params(params)
        report = run_backtest(config, klines_data, starting_equity=10000.0)

        for idx, trade in enumerate(report.trades):
            # No live OMLX decision exists in this fast screening pass —
            # confidence/recommendation stay fixed placeholders, same as
            # before. What changed is where was_profitable/pnl_percent
            # come from: a real BacktestTrade (real stop-loss/take-
            # profit/liquidation exit), not the next-candle-high formula.
            decision_id = f"basic_{params.name}_{idx}_{int(trade.entry_ts)}"
            self.accuracy_optimizer.record_trade_decision(
                decision_id=decision_id,
                confidence=75.0,
                recommendation="ADD_SAFETY",
                patterns_matched=[],
                entry_price=trade.entry_price,
            )
            self.accuracy_optimizer.record_trade_outcome(
                decision_id=decision_id,
                was_profitable=trade.is_real_win(),
                pnl_percent=trade.profit_percent,
                actual_bounce=trade.is_real_win(),
            )

        winning_trades = sum(1 for t in report.trades if t.is_real_win())
        losing_trades = len(report.trades) - winning_trades
        win_rate = winning_trades / len(report.trades) if report.trades else 0.0
        pnls = [t.profit_percent for t in report.trades]
        win_pnls = [t.profit_percent for t in report.trades if t.is_real_win()]
        loss_pnls = [t.profit_percent for t in report.trades if not t.is_real_win()]
        avg_win = statistics.mean(win_pnls) if win_pnls else 0.0
        avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0.0
        sharpe = 0.0
        if len(pnls) > 1:
            std_dev = statistics.stdev(pnls)
            sharpe = (statistics.mean(pnls) / std_dev) if std_dev > 0 else 0.0

        return ForwardTestResult(
            param_set=params,
            trades_count=len(report.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_percent=sum(pnls),
            avg_win_pnl=avg_win,
            avg_loss_pnl=avg_loss,
            max_drawdown=report.max_drawdown_percent,
            sharpe_ratio=sharpe,
            omlx_consulted=False
        )

    def _backtest_config_from_params(self, params: ParameterSet) -> "BacktestConfig":
        """Map ForwardOMLXTester's simplified 2-level ParameterSet onto a
        real backtest.py ladder. Confirmed with the user: treat the two
        levels as a real 2-rung ladder rather than expanding ParameterSet
        to backtest.py's full N-rung SEARCH_GRID shape (see
        run_everything.py's make_base_config for that fuller version).

        build_ladder()'s rung-1 deviation is dca_order_start_distance
        directly; rung-2's is dca_math.get_deviation_dca(step_percent,
        step_multiplier, n=2), which with step_multiplier=1 is exactly
        2 * step_percent — so step_percent = safety_trigger_2 / 2
        reproduces safety_trigger_2 as rung 2's actual distance from
        entry. safety_size_1/2 are treated as "safety order margin as %
        of a fixed base margin" — dca_order_size_multiplier is their
        ratio, matching build_ladder()'s prev_margin * multiplier rule
        for rung 2.
        """
        base_margin = 20.0  # matches run_everything.py's make_base_config convention
        return BacktestConfig(
            first_order_amount=base_margin,
            dca_order_amount=base_margin * params.safety_size_1 / 100,
            dca_max_order=2,
            dca_order_size_multiplier=(
                params.safety_size_2 / params.safety_size_1 if params.safety_size_1 else 1.0
            ),
            dca_order_start_distance=params.safety_trigger_1,
            dca_order_step_percent=params.safety_trigger_2 / 2,
            dca_order_step_percent_multiplier=1.0,
            dca_take_profit_percent=params.tp_percent,
            exchange_fee=0.06,
            leverage=params.leverage,
            auto_size_to_funds=False,
        )

    async def _test_params_with_omlx(
        self,
        params: ParameterSet,
        klines_data: list
    ) -> ForwardTestResult:
        """Test params WITH OMLX consulting every entry/exit, using
        backtest.py's real engine for the actual trade simulation (see
        _test_params' docstring for why the old next-candle-high formula
        was replaced). run_backtest() is a plain synchronous function
        and analyze_current_dip() below is async, and this method
        already runs inside its own event loop (via
        _run_forward_tester_cycle_sync's asyncio.run()) — nesting a
        second asyncio.run() inside a sync gate callback would raise
        "cannot be called from a running event loop". So this consults
        OMLX for every candle FIRST (unchanged from before — it already
        used static account_balance/position_size, never backtest.py's
        real evolving ladder state, so precomputing here isn't a
        fidelity regression), then hands the precomputed decisions to
        run_backtest() via a synchronous safety_order_gate that just
        looks up what OMLX said about the candle where a rung actually
        triggers.
        """
        decisions_by_bar: dict[int, dict] = {}

        for i in range(len(klines_data) - 1):
            kline = klines_data[i]
            entry_price = kline[4]
            entry_time = kline[0]

            # Feed candle to OMLX
            self.dip_service.update_with_candle(
                timestamp=int(entry_time),
                open_price=kline[1],
                high=kline[2],
                low=kline[3],
                close=kline[4],
                volume=kline[5] if len(kline) > 5 else 1000
            )

            # OMLX decision on entry
            if i == 0:
                self.dip_service.set_entry_price(entry_price)

            omlx_decision = await self.dip_service.analyze_current_dip(
                current_price=entry_price,
                account_balance=10000,
                position_size=params.base_order_size,
                leverage=params.leverage,
                tp_percent=params.tp_percent
            )

            # ML Enhancement: boost decisions with ML models
            ml_enhanced = False
            original_confidence = omlx_decision.confidence if omlx_decision else None
            original_decision = omlx_decision.reason if omlx_decision else None

            if omlx_decision and self.ml_advisor.models_available:
                # Pulled from omlx_decision.context (the real DipContext
                # BounceAnalyzer scored this dip on) and .dimension_scores
                # (BounceAnalyzer's own per-dimension 0-100 scores) —
                # previously every field below except entry_confidence/
                # leverage/candles_since_entry was a hardcoded constant
                # ("volume_ratio: 1.2, # Would come from actual data"),
                # meaning outcome_predictor.py's XGBoost model had almost
                # nothing that varied per trade to actually learn from.
                # context can be None on the emergency-bailout path (see
                # DrawdownDecision's docstring) — fall back to the same
                # neutral defaults extract_features_from_trade() itself
                # uses when a field is genuinely unavailable.
                ctx = omlx_decision.context
                dims = omlx_decision.dimension_scores
                trade_features = {
                    'volume_ratio': ctx.volume_ratio if ctx else 1.0,
                    'price_change_pct': (
                        (ctx.current_price - ctx.entry_price) / ctx.entry_price * 100
                        if ctx and ctx.entry_price else 0.0
                    ),
                    'momentum_score': ctx.rsi if ctx else 50.0,
                    'volatility_pct': (
                        ctx.atr / ctx.current_price * 100
                        if ctx and ctx.current_price else 1.0
                    ),
                    'spread_bps': ctx.bid_ask_spread_bps if ctx else 10.0,
                    'pattern_success_rate': dims.get('patterns', omlx_decision.confidence) if dims else omlx_decision.confidence,
                    'dip_depth_pct': ctx.dip_percent if ctx else 0.5,
                    'entry_confidence': omlx_decision.confidence,
                    'leverage': params.leverage,
                    'candles_since_entry': i,
                    'current_loss_pct': ctx.account_dd_percent if ctx else 0.0,
                }
                enhanced_decision, enhanced_confidence = self.ml_advisor.enhance_decision(
                    omlx_decision.reason,
                    omlx_decision.confidence,
                    trade_features
                )
                # Use enhanced values
                if enhanced_confidence > omlx_decision.confidence:
                    omlx_decision.confidence = enhanced_confidence
                    omlx_decision.reason = enhanced_decision
                    ml_enhanced = True

            decisions_by_bar[i] = {
                "decision": omlx_decision,
                "ml_enhanced": ml_enhanced,
                "original_confidence": original_confidence,
                "original_decision": original_decision,
            }

        def gate(ctx) -> bool:
            info = decisions_by_bar.get(ctx.bar_index)
            decision = info["decision"] if info else None
            if decision is None:
                return True  # no OMLX read for this bar — defer to the ladder's own behavior
            if decision.recommendation == "BAILOUT":
                return False
            # Same rule the old code applied to pnl AFTER the fact
            # ("confidence < 65: reduce or skip trade") — applied BEFORE
            # the fill actually happens instead.
            return decision.confidence >= 65

        config = self._backtest_config_from_params(params)
        report = run_backtest(config, klines_data, starting_equity=10000.0, safety_order_gate=gate)

        ts_to_bar = {kline[0]: idx for idx, kline in enumerate(klines_data)}
        omlx_trades_local = []
        for idx, trade in enumerate(report.trades):
            info = decisions_by_bar.get(ts_to_bar.get(trade.entry_ts))
            decision = info["decision"] if info else None

            omlx_trades_local.append(OMLXTrade(
                entry_price=trade.entry_price,
                entry_time=trade.entry_ts,
                exit_price=trade.exit_price,
                exit_time=trade.exit_ts,
                pnl_percent=trade.profit_percent,
                omlx_decision=decision.reason if decision else None,
                omlx_confidence=decision.confidence if decision else None,
                was_profitable=trade.is_real_win(),
                ml_enhanced=info["ml_enhanced"] if info else False,
                original_confidence=info["original_confidence"] if info else None,
                original_decision=info["original_decision"] if info else None,
            ))

            # See the matching comment in _test_params — same id-collision
            # bug, same fix (params.name disambiguates across the
            # parameter sets replayed over this klines_data).
            if decision:
                decision_id = f"omlx_{params.name}_{idx}_{int(trade.entry_ts)}"
                self.accuracy_optimizer.record_trade_decision(
                    decision_id=decision_id,
                    confidence=decision.confidence,
                    recommendation=decision.recommendation,
                    patterns_matched=decision.matched_patterns,
                    entry_price=trade.entry_price,
                )
                self.accuracy_optimizer.record_trade_outcome(
                    decision_id=decision_id,
                    was_profitable=trade.is_real_win(),
                    pnl_percent=trade.profit_percent,
                    actual_bounce=trade.is_real_win(),
                )

                # Feed the SAME calibration engine dca_bot.py's live/paper
                # engine records to, tagged so the dashboard can show
                # walk-forward decisions distinctly from live-paper ones
                # while both genuinely inform the same pattern/dimension
                # calibration.
                ctx = decision.context
                dip_record = self.calibration.record_decision(
                    entry_price=trade.entry_price,
                    dip_depth_percent=ctx.dip_percent if ctx else 0.0,
                    decision_confidence=decision.confidence,
                    decision_action=decision.recommendation,
                    patterns_matched=decision.matched_patterns,
                    dimension_scores=decision.dimension_scores or {},
                    source="walk_forward",
                )
                self.calibration.record_outcome(
                    dip_record=dip_record,
                    bounced=trade.is_real_win(),
                    max_depth_percent=abs(trade.raw_move_percent),
                    recovery_candles=max(
                        0, ts_to_bar.get(trade.exit_ts, 0) - ts_to_bar.get(trade.entry_ts, 0)
                    ),
                    safety_orders_needed=trade.safety_orders_used,
                    pnl=trade.profit_percent,
                )

        # Update global trades for learning
        self.omlx_trades.extend(omlx_trades_local)

        winning_trades = sum(1 for t in report.trades if t.is_real_win())
        losing_trades = len(report.trades) - winning_trades
        win_rate = winning_trades / len(report.trades) if report.trades else 0.0
        pnls = [t.profit_percent for t in report.trades]
        win_pnls = [t.profit_percent for t in report.trades if t.is_real_win()]
        loss_pnls = [t.profit_percent for t in report.trades if not t.is_real_win()]
        avg_win = statistics.mean(win_pnls) if win_pnls else 0.0
        avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0.0
        sharpe = 0.0
        if len(pnls) > 1:
            std_dev = statistics.stdev(pnls)
            sharpe = (statistics.mean(pnls) / std_dev) if std_dev > 0 else 0.0

        return ForwardTestResult(
            param_set=params,
            trades_count=len(report.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_percent=sum(pnls),
            avg_win_pnl=avg_win,
            avg_loss_pnl=avg_loss,
            max_drawdown=report.max_drawdown_percent,
            sharpe_ratio=sharpe,
            omlx_consulted=True
        )

    async def _omlx_learn_from_iteration(self, results: list[ForwardTestResult]) -> None:
        """OMLX learns from this iteration's results."""

        logger.info(f"\nOMLX Learning Phase:")

        # Analyze which decisions led to wins vs losses
        ml_enhanced_wins = 0
        ml_enhanced_losses = 0
        omlx_only_wins = 0
        omlx_only_losses = 0

        for trade in self.omlx_trades[-100:]:  # Last 100 trades
            if trade.was_profitable is None:
                continue

            # Track ML-enhanced vs OMLX-only outcomes
            if trade.ml_enhanced:
                if trade.was_profitable:
                    ml_enhanced_wins += 1
                else:
                    ml_enhanced_losses += 1
            else:
                if trade.was_profitable:
                    omlx_only_wins += 1
                else:
                    omlx_only_losses += 1

            # Record decision outcome
            if trade.omlx_decision and trade.omlx_confidence:
                ml_label = " [ML]" if trade.ml_enhanced else ""
                outcome_pattern = f"Conf{int(trade.omlx_confidence)}_{'WIN' if trade.was_profitable else 'LOSS'}{ml_label}"
                logger.debug(f"  {outcome_pattern}: {trade.omlx_decision}")

        # Update OMLX weights based on correlation
        winning_trades = [t for t in self.omlx_trades if t.was_profitable]
        losing_trades = [t for t in self.omlx_trades if not t.was_profitable]

        if winning_trades and losing_trades:
            avg_win_confidence = statistics.mean(
                [t.omlx_confidence for t in winning_trades if t.omlx_confidence]
            ) if any(t.omlx_confidence for t in winning_trades) else 0
            avg_loss_confidence = statistics.mean(
                [t.omlx_confidence for t in losing_trades if t.omlx_confidence]
            ) if any(t.omlx_confidence for t in losing_trades) else 0

            logger.info(f"  Winning trades avg confidence: {avg_win_confidence:.0f}%")
            logger.info(f"  Losing trades avg confidence: {avg_loss_confidence:.0f}%")
            logger.info(f"  Confidence delta: {avg_win_confidence - avg_loss_confidence:+.0f}%")

        # ML Enhancement Impact Analysis
        if ml_enhanced_wins + ml_enhanced_losses > 0:
            ml_win_rate = ml_enhanced_wins / (ml_enhanced_wins + ml_enhanced_losses)
            logger.info(f"\n  ML Enhancement Impact:")
            logger.info(f"    ML-Enhanced: {ml_win_rate:.1%} win rate ({ml_enhanced_wins}W/{ml_enhanced_losses}L)")

        if omlx_only_wins + omlx_only_losses > 0:
            omlx_win_rate = omlx_only_wins / (omlx_only_wins + omlx_only_losses)
            logger.info(f"    OMLX-Only: {omlx_win_rate:.1%} win rate ({omlx_only_wins}W/{omlx_only_losses}L)")

        if ml_enhanced_wins + ml_enhanced_losses > 0 and omlx_only_wins + omlx_only_losses > 0:
            ml_win_rate = ml_enhanced_wins / (ml_enhanced_wins + ml_enhanced_losses)
            omlx_win_rate = omlx_only_wins / (omlx_only_wins + omlx_only_losses)
            improvement = ml_win_rate - omlx_win_rate
            logger.info(f"    ML Improvement: {improvement:+.1%}")

        # Export learnings
        if len(self.omlx_trades) > 50:
            self.calibration.export_calibration_data(
                f"forward_test_calibration_{int(time.time())}.json"
            )
            self.accuracy_optimizer.export_accuracy_report(
                f"forward_test_accuracy_{int(time.time())}.json"
            )

            # Walk-forward training: retrain ML models from all collected data
            logger.info("\n" + "=" * 80)
            logger.info("Walk-Forward Training: Updating ML models from test data...")
            logger.info("=" * 80)
            try:
                if continuous_walk_forward_training():
                    logger.info("✓ Models retrained successfully")
                    # Reload ML advisor with new models
                    self.ml_advisor = get_ml_advisor()
                else:
                    logger.warning("⚠ Walk-forward training incomplete")
            except Exception as e:
                logger.error(f"Walk-forward training error: {e}")

        # Print accuracy summary
        logger.info("\n" + "=" * 80)
        self.accuracy_optimizer.print_summary()
        logger.info("=" * 80)

    def _get_top_5_params(self, results: list[ForwardTestResult]) -> list:
        """Get top 5 parameter sets by win rate and PnL."""
        # Sort by win rate first, then PnL
        sorted_results = sorted(
            results,
            key=lambda r: (r.win_rate, r.total_pnl_percent),
            reverse=True
        )
        return [(r.param_set, r) for r in sorted_results[:5]]

    def _generate_report(self) -> dict:
        """Generate comprehensive report."""
        accuracy_report = self.accuracy_optimizer.get_accuracy_report()

        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": self.duration_minutes,
            "iterations": self.iterations,
            "total_tests": len(self.all_results),

            "overall_statistics": {
                "best_win_rate": max((r.win_rate for r in self.all_results), default=0),
                "best_pnl": max((r.total_pnl_percent for r in self.all_results), default=0),
                "avg_win_rate": statistics.mean([r.win_rate for r in self.all_results]) if self.all_results else 0,
                "avg_pnl": statistics.mean([r.total_pnl_percent for r in self.all_results]) if self.all_results else 0,
            },

            "accuracy_metrics": accuracy_report,

            "omlx_learning": {
                "trades_with_omlx": len(self.omlx_trades),
                "omlx_wins": len([t for t in self.omlx_trades if t.was_profitable]),
                "omlx_losses": len([t for t in self.omlx_trades if not t.was_profitable]),
                "omlx_win_rate": len([t for t in self.omlx_trades if t.was_profitable]) / len(self.omlx_trades) if self.omlx_trades else 0,
                "avg_omlx_confidence": statistics.mean(
                    [t.omlx_confidence for t in self.omlx_trades if t.omlx_confidence]
                ) if any(t.omlx_confidence for t in self.omlx_trades) else 0,
            },

            "iteration_progression": self.iteration_metrics,

            "calibration_summary": {
                "patterns_learned": len(self.calibration.pattern_success_rates),
                "dimension_accuracy": self.calibration.dimension_accuracy,
                "recommended_weights": self.calibration.recommended_weights,
            }
        }

        return report


async def run_forward_tester(klines_data: list, duration_minutes: int = 30) -> dict:
    """Run forward OMLX tester.

    Args:
        klines_data: Historical 1-minute candle data
        duration_minutes: How long to run (default 30 minutes)

    Returns:
        Full testing report with OMLX learnings
    """
    tester = ForwardOMLXTester(duration_minutes=duration_minutes)
    return await tester.run(klines_data)
