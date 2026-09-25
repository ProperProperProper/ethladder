"""Test DIP analysis integration into dca_bot.py"""

import pytest
import asyncio
from symbot_python.strategy.market_data_collector import MarketDataCollector, OHLCV
from symbot_python.strategy.omlx_bounce_analyzer import BounceAnalyzer
from symbot_python.exchange.omlx_drawdown_advisor import OMLXDrawdownAdvisor
from symbot_python.exchange.dip_analysis_service import DipAnalysisService
from symbot_python.exchange.dip_calibration_engine import DipCalibrationEngine


def test_market_data_collector_calculates_indicators():
    """Verify MarketDataCollector calculates all indicators."""
    collector = MarketDataCollector()

    # Feed 50 candles
    for i in range(50):
        candle = OHLCV(
            timestamp=1000 + i,
            open=3000 + i,
            high=3001 + i,
            low=2999 + i,
            close=3000.5 + i,
            volume=1000 + i * 10
        )
        collector.update(candle)

    # Should now have all indicators ready
    assert collector.is_ready()
    assert collector.get_rsi() is not None
    assert collector.get_atr() is not None
    assert collector.get_macd_histogram() is not None
    assert collector.get_stochastic_k() is not None
    bb = collector.get_bollinger_bands()
    assert bb is not None
    assert len(bb) == 3


def test_bounce_analyzer_analyzes_dip():
    """Verify BounceAnalyzer creates analysis from market context."""
    from symbot_python.strategy.omlx_bounce_analyzer import DipContext

    analyzer = BounceAnalyzer()

    # Create a dip context
    context = DipContext(
        entry_price=3000,
        current_price=2985,  # -0.5% dip
        dip_percent=0.5,
        dip_candles=1,
        current_volume=1500,
        avg_volume_20=1000,
        volume_ratio=1.5,
        recent_high=3010,
        recent_low=2980,
        wick_formed=True,
        candle_closing_up=True,
        rsi=28,  # Oversold
        macd_histogram=0.1,  # Turning
        stochastic_k=18,  # Oversold
        atr=5,
        atr_avg_20=4,
        bollinger_high=3020,
        bollinger_low=2980,
        bollinger_pct_b=0.1,
        time_utc=15,
        day_of_week=3,
        bitcoin_trend=1,
        bid_ask_spread_bps=2.0,
        account_dd_percent=5.5,  # 0.5% * 11
        liquidation_buffer=94.5
    )

    # Analyze
    analysis = analyzer.analyze_dip(context)

    assert analysis is not None
    assert 0 <= analysis.probability <= 100
    assert analysis.recommendation in [
        "ADD_SAFETY_AGGRESSIVE",
        "ADD_SAFETY",
        "WAIT_1_CANDLE",
        "BAILOUT"
    ]


async def test_dip_analysis_service_detects_dips():
    """Verify DipAnalysisService detects and analyzes dips."""
    service = DipAnalysisService()

    # Feed 50 candles
    for i in range(50):
        service.update_with_candle(
            timestamp=1000 + i,
            open_price=3000 + i,
            high=3001 + i,
            low=2999 + i,
            close=3000.5 + i,
            volume=1000 + i * 10
        )

    # Set entry
    service.set_entry_price(3000)

    # Simulate a dip
    decision = await service.analyze_current_dip(
        current_price=2985,  # -0.5% dip
        account_balance=10000,
        position_size=1.0,
        leverage=11,
        tp_percent=0.33
    )

    assert decision is not None
    assert decision.confidence >= 0
    # Should have a decision about whether to add safety
    assert isinstance(decision.should_add_safety, bool)


def test_calibration_engine_records_and_calibrates(tmp_path):
    """Verify DipCalibrationEngine records decisions and outcomes."""
    calibration = DipCalibrationEngine(state_file=str(tmp_path / "calibration_state.json"))

    # Record a decision
    record = calibration.record_decision(
        entry_price=3000,
        dip_depth_percent=0.5,
        decision_confidence=85,
        decision_action="ADD_SAFETY",
        patterns_matched=["support_bounce", "rsi_oversold"],
        dimension_scores={
            "volume": 75,
            "price_action": 80,
            "momentum": 90,
            "volatility": 70,
            "risk": 80
        }
    )

    assert record is not None

    # Record outcome
    calibration.record_outcome(
        dip_record=record,
        bounced=True,
        max_depth_percent=0.7,
        recovery_candles=2,
        safety_orders_needed=1,
        pnl=7.26  # 0.66% * 11
    )

    # Should have recorded calibration data
    assert len(calibration.trades) == 1
    assert calibration.trades[0].outcome_bounced == True


async def test_dip_advisor_makes_decisions():
    """Verify OMLXDrawdownAdvisor makes sound decisions."""
    advisor = OMLXDrawdownAdvisor()

    decision = await advisor.analyze_drawdown(
        entry_price=3000,
        current_price=2985,  # -0.5% dip
        recent_high=3010,
        recent_low=2980,
        atr=5,
        atr_avg_20=4,
        rsi=28,  # Oversold
        macd_histogram=0.1,
        stochastic_k=18,
        current_volume=1500,
        avg_volume_20=1000,
        current_time_utc=15,
        day_of_week=3,
        bitcoin_trend=1,
        bid_ask_spread_bps=2.0,
        account_dd_percent=5.5,
        liquidation_buffer=94.5,
        bollinger_high=3020,
        bollinger_low=2980
    )

    assert decision is not None
    assert isinstance(decision.should_add_safety, bool)
    assert isinstance(decision.confidence, (int, float))
    assert decision.liquidation_risk in ["LOW", "MEDIUM", "HIGH"]


# Run async tests
@pytest.mark.asyncio
async def test_full_integration_flow(tmp_path):
    """Test complete flow: feed data → analyze → decide → calibrate."""
    service = DipAnalysisService()
    calibration = DipCalibrationEngine(state_file=str(tmp_path / "calibration_state.json"))

    # Feed initial data
    for i in range(50):
        service.update_with_candle(
            timestamp=1000 + i,
            open_price=3000 + i,
            high=3001 + i,
            low=2999 + i,
            close=3000.5 + i,
            volume=1000 + i * 10
        )

    # Set entry
    service.set_entry_price(3000)

    # Analyze dip
    decision = await service.analyze_current_dip(
        current_price=2985,
        account_balance=10000,
        position_size=1.0,
        leverage=11,
        tp_percent=0.33
    )

    # Record decision
    if decision:
        # Determine action from decision
        if decision.should_add_safety:
            action = "ADD_SAFETY"
        else:
            action = "BAILOUT"

        record = calibration.record_decision(
            entry_price=3000,
            dip_depth_percent=0.5,
            decision_confidence=decision.confidence,
            decision_action=action,
            patterns_matched=["support_bounce"],
            dimension_scores={"volume": 75, "momentum": 90, "price_action": 80}
        )

        # Simulate trade completing
        calibration.record_outcome(
            dip_record=record,
            bounced=True,
            max_depth_percent=0.7,
            recovery_candles=2,
            safety_orders_needed=1,
            pnl=7.26
        )

        # Verify calibration captured the trade
        assert len(calibration.trades) >= 1
        assert calibration.trades[-1].outcome_bounced == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
