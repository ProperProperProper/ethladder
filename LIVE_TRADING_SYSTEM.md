# Live Trading System - Complete Documentation

## 🎯 Overview

ETH Ladder v2.0.5 integrates real Bybit mainnet trading with automated OMLX learning. This document teaches the learning bridges (Claude, OMLX, Codex) how the live trading system works.

## 🔴 Live Trading Activation

### Manual Control Only
- **No auto-start** — User must click "▶ Start Trading" on dashboard
- **Auto-detection** — If positions detected on Bybit, manager auto-initializes
- **Graceful shutdown** — Click "⏹ Stop Trading" to pause (positions stay open)

### Credential Management
- **Storage**: macOS Keychain (service: "unified-combo-grid", account: "live")
- **Security**: Never logged, never exposed in code/files/exports
- **Auto-read**: Factory reads from Keychain when initializing
- **Dashboard UI**: Password manager in System tab (Save/Clear buttons)

## 💰 Account & Balance

### Real Money
- **Balance source**: Bybit unified account USDT balance
- **Minimum**: $1.00 USDT (below that, paper trading fallback $200)
- **Current**: $117.76 USDT (verified 2025-09-25 13:45 UTC)
- **Cache**: Populated on `verify_connection()` via `get_balance("USDT")`

### Margin Tracking
```python
margin_balance = <current USDT balance from Bybit>
```
Updated every 30-60 seconds during equity sampling loop.

## 📊 Equity Sampling

### What It Tracks
```
Total Equity = Start Balance + Realized P&L + Unrealized P&L

Where:
  Realized   = Current Balance - Session Start Balance
  Unrealized = Position Qty × (Current Price - Entry Price) - Funding Costs
```

### Implementation
- **Loop**: Runs every 60 seconds (if trading_enabled)
- **Sync reading**: Uses `position()` method (cached, non-blocking)
- **Handles edge cases**:
  - No position open → unrealized = 0
  - No balance change → realized = 0
  - Funding costs deducted from unrealized

### Data Flow
1. Manager initialized with real Bybit balance
2. `verify_connection()` calls `get_balance("USDT")` → caches margin_balance
3. `position()` returns cached position from Bybit
4. Equity loop samples every 60 seconds: `margin_balance + position.margin_committed`
5. History recorded to `_equity_history` (in-memory, up to 1000 samples)

## 🎯 Order Execution

### Two Methods Available

**Market Orders (0.06% taker fee)**
```python
await manager.exchange.place_market_order("ETHUSDT", "BUY", qty, price)
```
- Fills immediately at market price
- Higher slippage cost
- Used for emergency closes

**Limit Orders (0.01% maker fee - 85% savings)**
```python
await manager.exchange.place_limit_order("ETHUSDT", "BUY", qty, price)
```
- Uses Bybit `timeInForce="PostOnly"` (native parameter)
- Fills at limit price or better
- Fee efficient for planned entries
- Can fail to fill if price moves away

### Order Precision
- Quantity: 0.01 ETH minimum step
- Price: $0.01 USDT minimum tick
- Both enforced by `InstrumentPrecision` checks

### Reduce-Only Enforcement
- Buy orders when short → `reduce_only=true`
- Sell orders when long → `reduce_only=true`
- Prevents position-flipping accidents

## 🛡️ Safety Guards

### Leverage Policy (non-negotiable)
- **Range**: 9-11x (hardcoded in `leverage_policy.py`)
- **Enforcement**: Set before each order via `ensure_leverage()`
- **Auto-recovery**: Never exceeds configured bounds

### Position Sizing
- **Funds utilization**: 78% (configurable 25-98%)
- **Dynamic sizing**: Adjusted based on available balance
- **Minimum**: 25% floor (can't trade less than 25% utilization)

### Liquidation Modeling
- **Backtest simulations**: Account for liquidation risk
- **Live protection**: Maintain 2x buffer above liquidation price
- **Monitoring**: Continuous tracking during forward testing

## 📡 Dashboard Integration

### Live Trading Tab
- **Status badge**: 🔴 LIVE (green when trading_enabled=true)
- **Buttons**:
  - ▶ Start Trading (initialize manager, enable trading)
  - ⏹ Stop Trading (disable trading, keep positions)
  - ✖ Close All Positions (market orders to exit)
- **Updates**: Every 5 seconds via `/live/status` endpoint

### Position Tracking
- **Paper positions**: 📄 badge (PaperExchangeClient)
- **Live positions**: 🔴 badge (BybitClient)
- **Unrealized P&L**: Calculated per position
- **Updated**: Every 30 seconds via `export_positions()`

### Credentials Card
- **Status**: Green ✓ when credentials saved
- **Save**: Submit API Key + Secret → Keychain
- **Clear**: Remove credentials → Keychain

## 🔗 Learning Bridge Integration

### For Claude (via Codex MCP Bridge)
```
@ask "get_ethladder_state"
→ Returns: live_status, positions (paper + live), system_metrics, equity_tracking
```

### For OMLX (via CLI Bridge)
```bash
./ethladder_omlx_bridge.py "What's the current live trading status?"
→ Analyzes: paper vs live positions, equity sampling, system health
```

### For Claude CLI
```bash
./ethladder_cli_bridge.py state
→ Shows: position counts (paper/live), system metrics, live margin balance
```

## 📈 OMLX Integration

### How OMLX Learns from Live
1. **Real trade outcomes**: Recorded to `dip_calibration_state.json`
2. **Dimension updates**: 10-D bounce probability recalibrated
3. **Success rates**: Pattern accuracy tracked across live trades
4. **Retraining**: ML models update every 2 minutes using live data

### Key Metrics Available
- `bounce_probability` (0-100%): Current DIP recovery likelihood
- `dimension_scores`: 10 dimensions with predictiveness %
- `pattern_success_rates`: morning_dip, support_bounce accuracy
- `training_events_count`: Total calibration updates

## 🔄 Data Persistence

### Core Files
- `positions.json`: Paper + live positions (30s refresh)
- `omlx_metrics_live.json`: Current OMLX state (1min refresh)
- `ml_trainer_status.json`: Model training progress (2min refresh)
- `live_status.json`: Trading mode, equity history (60s refresh)
- `ethladder_analytics.db`: SQLite persistence (all metrics)

### Atomic Writes
- All JSON writes use temp file + `os.replace()` for crash safety
- Trade memory capped at 20,000 entries (auto-dedup by filename)
- Database uses `check_same_thread=False` for async compatibility

## ⚠️ Known Limitations

1. **No position sync on restart**: Live position state not persisted to disk (would require separate API call on startup)
2. **Equity history in-memory only**: Lost on restart (re-samples from Bybit on next start)
3. **Order fill detection**: Market orders assume 100% fill (actual fills checked via position queries)

## 🚀 Common Workflows

### Starting Live Trading
```
1. User clicks "▶ Start Trading" on dashboard
2. Dashboard calls POST /live/start
3. live_trading.start_live_trading() called:
   - Sets _trading_enabled = True
   - Calls get_manager() → initializes with real balance
   - Starts equity sampling loop
4. Manager ready to place orders
```

### Placing a Live Order
```
1. DCA strategy decides to enter
2. Calls place_limit_order(..., timeInForce="PostOnly")
3. Bybit API executes (0.01% maker fee)
4. Position updated in `position_cache`
5. Equity sampling picks up new position next cycle
6. OMLX recalibrates bounce probability with new data
```

### Handling a Position
```
1. Entry order fills → position tracked
2. Equity loop samples: margin_balance + margin_committed
3. Price updates every candle
4. Take-profit/liquidation checked continuously
5. Exit order placed → position closed
6. Trade recorded to trade_memory.json + dip_calibration_state.json
7. ML models retrain (next 2-min cycle)
```

## 🔍 Troubleshooting

### "Trading says already running but I didn't click Start"
- **Cause**: check_open_positions() detected a position on Bybit
- **Fix**: Close position on Bybit, then click Start again

### "Equity showing as $0"
- **Cause**: margin_balance_cache not populated (get_balance not called)
- **Fix**: verify_connection() now ensures cache is populated
- **Check**: `manager.exchange.margin_balance` should show current USDT

### "Orders not executing"
- **Cause**: Trading disabled, or insufficient balance
- **Fix**: Click Start Trading, verify balance >= $1
- **Debug**: Check `/live/status` endpoint for manager state

### "No positions showing in dashboard"
- **Cause**: export_positions() runs every 30 seconds, may be stale
- **Fix**: Wait 30 seconds and refresh, or click a button to trigger refresh

## 📚 See Also

- `CLAUDE.md` — Architecture and constraints
- `FINE_TUNING_GUIDE.md` — ML training data sources
- `LEARNING_BRIDGES_README.md` — How to query bot data
- `TESTING_POLICY.md` — Safety and validation rules

## 🔄 Recent Changes (v2.0.5)

**Fixed:**
- Added `margin_balance` property to BybitClient (was missing)
- verify_connection() now populates cache on init
- _sample_equity handles None position without crashing

**Improved:**
- Equity sampling now fully operational
- Real Bybit balance tracked and cached
- Dashboard can monitor session P&L

**Current Status:** ✅ Live trading system fully operational and ready for user-driven orders.
