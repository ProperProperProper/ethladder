# ETH Ladder Fine-Tuning & Learning Guide

This guide explains how to leverage the bot's collected training data and OMLX learning system for model improvement and strategy refinement.

## 🤖 Codex Integration

The bot exports live data via an MCP bridge (`ethladder_bot_mcp.py`) that allows Codex/Claude to query:

### Available Data Endpoints

1. **Bot State** (`get_ethladder_state`)
   - Current positions (paper & live)
   - Trading mode (disabled/live)
   - System metrics (CPU, memory, disk)
   - Available balance

2. **OMLX Learning** (`get_omlx_learning`)
   - 10-dimension bounce probability analysis
   - DIP detection calibration (5,000 training events)
   - Pattern success rates (morning_dip, support_bounce)
   - Decision outcomes and confidence levels

3. **ML Training** (`get_ml_training`)
   - XGBoost outcome predictor status
   - RL decision optimizer Q-table state
   - Trade memory (last 20,000 trades, capped)
   - Current win rate and model accuracy

4. **Walk-Forward Results** (`get_walk_forward_results`)
   - Best parameter set (14-day backtest validated)
   - Profit factor and max drawdown
   - Out-of-sample performance metrics
   - Parameter space search coverage

5. **Complete Performance Snapshot** (`get_trading_performance`)
   - All of the above combined

## 📊 Training Data Sources

### Real Trade Data

**Location:** `trade_memory.json` (20,000 cap, deduped)

Contains every closed deal with:
- Entry/exit prices and time
- P&L (absolute and %)
- Win/loss classification
- Confidence level
- Pattern type and action taken

**Updated by:**
- Forward tester (every 30 min, walk-forward backtests)
- Live/paper trading (every deal close)

### OMLX Calibration Data

**Location:** `dip_calibration_state.json`

Contains:
- 5,000 historical DIP decisions with outcomes
- 10 dimensions with predictiveness scores (100% = perfect)
- Pattern-specific success rates
- Time-based outcome latency (how long recovery takes)

**Dimensions tracked:**
- Correlation, Risk, Volume, Volatility
- Technical Setup, Price Action, Patterns
- Microstructure, Momentum, Time

### ML Training Cycles

**Location:** `ml_trainer_status.json`

Tracks:
- Training cycles (every 2 minutes)
- Model retraining iterations
- Accuracy improvement over time
- Feature importance (which inputs drive predictions)

### Walk-Forward Validation

**Location:** `walk_forward_training_summary.json`

Latest backtest results:
- Parameter ranges tested (78-100% funds utilization)
- Profit factor by parameter
- Win rate by parameter
- Recommended leverage and position sizing

## 🎯 How to Use for Fine-Tuning

### Pattern Analysis

```
Query: get_omlx_learning
Use: Analyze which dimensions predict bounces best
Goal: Improve entry signal confidence threshold
```

### Model Improvement

```
Query: get_ml_training
Use: Review recent trade outcomes and features
Goal: Retrain XGBoost with latest market data patterns
```

### Parameter Validation

```
Query: get_walk_forward_results
Use: Compare current best params vs new candidates
Goal: Verify new leverage/sizing rules work live
```

### Performance Diagnosis

```
Query: get_trading_performance
Use: Full system health check and bottleneck analysis
Goal: Identify which subsystem (OMLX/ML/params) needs tuning
```

## 📈 Current Performance Baseline

**From latest walk-forward:**
- Win Rate: 70%
- Profit Factor: 1.35x
- Max Drawdown: ~8%
- Recovery Success: 100%
- Funds Utilization: 78% (conservative)

**OMLX Accuracy:**
- morning_dip: 59.4% (503 samples)
- support_bounce: 86.3% (2,605 samples)

**Trade Memory:**
- Total trades: 20,000 (capped)
- P&L range: -2.2% to +2.97% per trade
- Steady state achieved

## 🔄 Training Loop

1. **Live trading** generates real trade outcomes
2. **Trade memory** accumulates (deduplicated, 20k cap)
3. **ML trainer** retrains XGBoost every 2 min
4. **OMLX engine** calibrates bounce dimensions continuously
5. **Walk-forward tester** validates params every 30 min
6. **Best params** sync to live bot every 8 hours

## ⚠️ Important Notes

- **Never modify** bot state from Codex (read-only)
- **API keys stored in Keychain** (never exposed in data)
- **Real money trades** — validation required before deployment
- **14-day backtest window** (non-negotiable for safety)
- **Leverage: 9-11x bounded band** (fixed risk model)

## 🚀 Next Steps

1. Use Codex to query bot performance data
2. Identify improvement opportunities (pattern, model, params)
3. Test changes in walk-forward mode (30-min cycles)
4. Validate on live paper trading (no real funds)
5. Deploy to live trading only after validation

## 📝 Version Info

- Bot Version: 2.0+ (Live trading enabled)
- OMLX Dimensions: 10 (full ensemble)
- ML Models: XGBoost + RL Q-Learning
- Walk-Forward Cycles: Continuous (14-day windows)
- Trade Memory: 20,000 cap (atomic writes, deduplicated)
