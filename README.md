# 🪜 ETH Ladder - Unified Trading System

<img width="1451" height="621" alt="Screenshot 2026-09-25 at 5 42 16 PM" src="https://github.com/user-attachments/assets/502b4b5c-1a7f-4328-aab0-13259f0bcd32" />
<img width="1442" height="722" alt="Screenshot 2026-09-25 at 5 42 39 PM" src="https://github.com/user-attachments/assets/4273f710-d64f-49d0-8989-c54c334b1ea2" />
<img width="1449" height="664" alt="Screenshot 2026-09-25 at 5 42 48 PM" src="https://github.com/user-attachments/assets/b1fbe7a8-d082-492c-93bb-9c6f20310bfc" />

**Production-Ready Self-Improving Trading Bot with Real-Time Dashboard & ML Learning**

## 🎯 What This Is

A complete autonomous trading system, running as **one script**
(`run_everything.py`), that:
- ✅ Continuously learns from trading data (OMLX + ML)
- ✅ Auto-restarts on crash via launchd
- ✅ Persists all data to SQLite for historical analysis
- ✅ Shows everything on a single unified dashboard
- ✅ Stays under 80% CPU via built-in throttling

## 🚀 Quick Start

**Everything — web server, dashboard, paper trading bot, forward tester,
continuous optimizer, ML trainer, data exporters, SQLite saver, resource
monitor, data pruner, log watcher — is ONE script, ONE process, ONE PID:
`run_everything.py`.** There is nothing else to run.

**Runs automatically via launchd** (`com.ethladder.unified`, installed at
`~/Library/LaunchAgents/com.ethladder.unified.plist`) — starts when you
log into this Mac, restarts itself if it ever crashes. Nothing to run
manually in normal use.

```bash
# Check it's running:
launchctl list | grep ethladder
curl http://127.0.0.1:8731/console

# Open dashboard:
open http://127.0.0.1:8731/console

# Restart it (e.g. after pulling code changes):
launchctl unload ~/Library/LaunchAgents/com.ethladder.unified.plist
launchctl load ~/Library/LaunchAgents/com.ethladder.unified.plist

# Stop it entirely (survives until you launchctl load again or reboot):
launchctl unload ~/Library/LaunchAgents/com.ethladder.unified.plist

# Run it manually instead (foreground, e.g. for debugging — first unload
# the launchd job above or they'll fight over port 8731):
.venv/bin/python run_everything.py
```

Uses the project's `.venv` (not the system `python3` — `uvicorn` and the
other dependencies are only installed there; `python3 run_everything.py`
against a bare system Python will fail with `ModuleNotFoundError`).

**Note:** a LaunchAgent starts when *you log into your macOS user
account*, not at power-on before login. If this Mac doesn't auto-login,
the bot starts the moment you log in, not before.

## 📊 Dashboard

**Single Unified Master Dashboard** at `http://127.0.0.1:8731/console`

### 8 Tabs:

1. **🧠 OMLX Learning** - ML metrics, patterns, training cycles
2. **📄 Paper Trading** - Simulated trades, recent results, P&L
3. **📊 Backtest Results** - Performance metrics, Sharpe ratio, profit factor
4. **📍 Position Tracking** - Paper & live open positions, entry/exit, unrealized P&L
5. **🔴 Live Trading** - Real Bybit orders, Start/Stop controls, equity tracking
6. **💰 Funds Utilization** - Automated position sizing analysis (78-100% range)
7. **📈 Analytics** - Comparison charts, performance trends, system health
8. **🖥️ System Status** - CPU, RAM, Disk, Temperature real-time + 🔐 API Credentials manager

**Updates every 15 seconds** from live JSON data.

### Live Trading Features
- **Manual control**: Start/Stop buttons (no auto-start unless position detected)
- **Real money**: Executes on Bybit mainnet with actual $117+ USDT balance
- **Fee efficient**: 85% savings via POST_ONLY limit orders (0.01% maker vs 0.06% taker)
- **Equity tracking**: Monitors session P&L from margin balance + position tracking
- **Credential manager**: Save/clear Bybit API keys via dashboard (stored in macOS Keychain)

## 🔄 Architecture — One Process, Nine Background Tasks

`run_everything.py` starts a FastAPI/uvicorn process containing:

- **Web server + dashboard** — the FastAPI app itself
- **Trading bot (paper)** — `symbot_python/api/paper.py`'s `DCABotManager`,
  driven by the app's own routes (this is what makes "the bot" and "the
  web server" the same process)

...plus 10 asyncio background tasks sharing that one process:

| Task | Purpose | Interval |
|------|---------|----------|
| `optimizer` | Walk-forward parameter search | Every 4h |
| `forward_tester` | Simulated trading / OMLX learning data | 30-min cycles |
| `ml_trainer` | XGBoost + Q-Learning retraining | Every 2 min |
| `data_exporter` | Paper/backtest/position JSON export | Every 30s |
| `omlx_metrics_exporter` | OMLX learning metrics for dashboard | Every 30s |
| `sqlite_saver` | Persist metrics to SQLite | Every 45s |
| `resource_monitor` | CPU/Memory/Disk/Temp export | Every 5s |
| `pruner` | Archive/VACUUM old DB rows | Once per day |
| `log_watcher` | Alert on ERROR/CRITICAL + tracebacks | Polls every 2s |
| `inbox_pattern_extractor` | Reviews the external training-inbox pipeline's approved Q&A for candidate DCA-strategy patterns (never auto-applied — dashboard promote/reject) | Every 1h |

Two things make sharing one process safe (see `run_everything.py`'s own
module docstring for the full rationale):

- **CPU-bound work never runs on the shared event loop.** The optimizer's
  walk-forward search and the forward tester's parameter-combination
  testing each run via `asyncio.to_thread()` wrapping a private
  `asyncio.run()` call, isolated on a worker thread.
- **Only one FileHandler is attached to the root logger** —
  `logs/run_everything.log` — so the log watcher's own alerts can never
  be written back into the file it's tailing (that caused a real
  110MB/minute feedback loop while this was being built).

## 💾 Data Persistence

### SQLite Database (`ethladder_analytics.db`)

**7 Tables:** `omlx_metrics`, `system_metrics`, `paper_trades`,
`backtest_results`, `positions`, `patterns`, `daily_summary`

- Saved every 45 seconds by the `sqlite_saver` task
- Pruned daily by the `pruner` task (30-day retention, VACUUM)
- Emergency cleanup triggers if free disk < 5GB

### Critical alerts also go to SQLite

`logs/watcher_alerts.log` and `data/watcher_alerts.db` — the `log_watcher`
task writes here for ERROR/CRITICAL/WARNING lines and Python tracebacks
only. Routine data (candles, walk-forward iteration output, test
results) stays in the plain log file — it never touches SQLite.

### Analysis Tools
```bash
# Query the database directly
sqlite3 ethladder_analytics.db "SELECT * FROM omlx_metrics ORDER BY timestamp DESC LIMIT 10;"

# Full historical analysis
python3 analyze_history.py
```

## 🧠 Learning System

**XGBoost Outcome Predictor** — predicts win/loss probability,
`omlx_outcome_model.pkl`

**RL Decision Optimizer** — Q-Learning over discretized market states,
`omlx_rl_model.json`

**Pattern Analyzer** — discovers successful market patterns, tracks
success rates

**Trade Memory** — `trade_memory.json`, hard-capped at 20,000 trades
(`symbot_python/ml/trade_memory.py`'s `TradeMemory.MAX_TRADES`), atomic
writes (temp file + `os.replace`), tracks which forward-test reports it's
already folded in so retraining never re-ingests the same data twice.

### Continuous Learning Loop
1. `forward_tester` task runs 30-min cycles
2. Trade memory accumulates outcomes (capped, deduped)
3. `ml_trainer` task retrains every 2 minutes
4. Models improve from new data
5. Dashboard shows progress in real time

## 🎓 Learning Bridges — Teach Claude, OMLX, and Codex

**Three bridges export bot data for LLM fine-tuning and autonomous analysis:**

### Claude CLI Bridge (`ethladder_cli_bridge.py`)
```bash
./ethladder_cli_bridge.py state          # Current positions, equity, system health
./ethladder_cli_bridge.py omlx           # 10-D bounce analysis and patterns
./ethladder_cli_bridge.py ml             # Model training status and trades
./ethladder_cli_bridge.py backtest       # Walk-forward validation results
./ethladder_cli_bridge.py watch --interval 5  # Live monitoring
./ethladder_cli_bridge.py export         # Full data export for fine-tuning
```

### Codex MCP Bridge (`Codex/.integrations/smart-model-router/ethladder_bot_mcp.py`)
```
@ask "get_ethladder_state"           # Query bot state in Codex
@ask "get_omlx_learning"             # Get OMLX metrics
@ask "get_ml_training"               # Get model training data
@ask "get_walk_forward_results"      # Get backtest validation
@ask "get_trading_performance"       # Complete snapshot
```

### OMLX Local Model Bridge (`Codex/.integrations/smart-model-router/ethladder_omlx_bridge.py`)
```bash
./ethladder_omlx_bridge.py                      # Interactive mode
./ethladder_omlx_bridge.py "What's the bounce?" # Query mode
```

**See [LEARNING_BRIDGES_README.md](LEARNING_BRIDGES_README.md) for full documentation.**

## ⚡ CPU Management

- `symbot_python/system/cpu_limiter.py` monitors real CPU usage
- 70%: optimizer waits before starting a new cycle
- 75%: exporters begin throttling (exponential backoff)
- 80%: hard limit — non-critical work is deferred to next cycle

## 🔐 Auto-Start via launchd, No Desktop Notifications

As of 2026-09-24 (reinstated same day after a brief manual-only period):
`com.ethladder.unified` is installed and loaded — auto-starts on login,
auto-restarts on crash (`KeepAlive`). The log watcher still does **not**
fire native macOS desktop notifications for ERROR/CRITICAL log lines
(that removal was not reversed — it still records every one to
`logs/watcher_alerts.log` and `data/watcher_alerts.db`, just without
interrupting the desktop). See "Quick Start" above for how to
stop/restart it.

## 📁 Directory Structure

```
ethladder/
├── run_everything.py               # THE ENTRY POINT — everything runs here
├── analyze_history.py              # Manual historical analysis (read-only)
│
├── symbot_python/                  # Library code — imported by run_everything.py
│   ├── api/                        #   FastAPI app, paper trading routes
│   ├── strategy/                   #   dca_bot.py, backtest.py, optimize.py,
│   │                                #   forward_omlx_tester.py, optimization_store.py
│   ├── exchange/                   #   OMLX dip analysis, keychain, Bybit clients
│   ├── ml/                         #   continuous_trainer.py, trade_memory.py,
│   │                                #   outcome_predictor.py, rl_decision_optimizer.py
│   ├── system/                     #   cpu_limiter.py, resource_monitor.py
│   └── signals/                    #   candle utilities
│
├── ethladder_master_dashboard.html # THE dashboard (served by run_everything.py)
├── ethladder_analytics.db          # SQLite database
├── trade_memory.json               # Trade history (capped, deduped)
├── omlx_rl_model.json              # RL Q-table
├── omlx_metrics_live.json          # Live OMLX metrics (dashboard reads this)
├── system_metrics.json             # Live CPU/RAM/Disk/Temp
│
├── logs/run_everything.log         # THE log file (everything writes here)
├── data/watcher_alerts.db          # SQLite alert log (critical issues only)
├── data/optimizer_status.json      # Live optimizer progress
│
├── scripts/launchd/com.ethladder.unified.plist  # The one launchd job
└── tests/                          # pytest suite
```

## ⚙️ Configuration

### Timings (all in `run_everything.py`, top-level constants)
- `OPTIMIZER_CYCLE_SECONDS` — 4 hours
- `FORWARD_TESTER_DURATION_MINUTES` / `FORWARD_TESTER_PAUSE_SECONDS` — 30 min / 15s
- `ML_TRAINER_CHECK_INTERVAL_SECONDS` — 2 min
- `DATA_EXPORT_INTERVAL_SECONDS` — 30s
- `SQLITE_SAVE_INTERVAL_SECONDS` — 45s
- `RESOURCE_MONITOR_INTERVAL_SECONDS` — 5s
- `PRUNER_INTERVAL_SECONDS` — 24h
- Dashboard refresh (browser-side) — 15s

### Web Server
- **Port:** 8731 (fixed)
- **Address:** http://127.0.0.1:8731/console

## 🔒 Non-Negotiable Trading Constraints

- **Leverage:** 9x–11x only (`symbot_python/strategy/leverage_policy.py`)
- **Instrument:** ETHUSDT only, Bybit mainnet only, never testnet
- **Account:** Real available balance only (read-only Keychain lookup —
  never places an order)
- **Backtest window:** Strict 14-day cap, always walk-forward validated
  (never a single-window backtest)

## 🐛 Troubleshooting

### Not starting?
```bash
launchctl unload ~/Library/LaunchAgents/com.ethladder.unified.plist
.venv/bin/python run_everything.py    # run in foreground, read the traceback directly
```

### Dashboard not loading?
```bash
curl http://127.0.0.1:8731/console
lsof -i :8731
```

### launchd job not running?
```bash
launchctl list | grep ethladder
tail -f /tmp/com.ethladder.unified.stdio.log
tail -f logs/run_everything.log
```

### Database growing too large?
The `pruner` task runs daily automatically. To force it, restart the
process (it prunes once at each daily interval, not on startup) or query
`ethladder_analytics.db` directly with `sqlite3 ... VACUUM`.

### Check logs
```bash
tail -f logs/run_everything.log
tail -f logs/watcher_alerts.log   # critical alerts only
```

## 📝 Recent Significant Changes

**2026-09-24 (later):** Auto-start reinstated — `com.ethladder.unified`
installed to `~/Library/LaunchAgents/`, loaded, verified running (port
8731, `/console` returns 200). Desktop notifications remain off. Also
did a full repo cleanup: untracked ~23.7M lines of runtime/scratch JSON
that had been accidentally committed (`trade_memory.json` had reached
1.3GB in history), stripped it out of git history entirely via
`git filter-repo` (force-pushed the rewritten history), and deleted 23
legacy pre-consolidation design docs that described scripts already
deleted in the merge below. Confirmed via full-history scan: no secrets,
API keys, or credentials anywhere in this repo, ever.

**2026-09-23:** Merged into one script (`run_everything.py`), old
fragmented scripts and launchd jobs deleted entirely. Fixed a real crash
cause found along the way: `trade_memory.json` had grown to 5.7GB from
an unbounded re-ingestion bug. See `CLAUDE.md`'s "Recent Significant
Changes" for the full list of fixes.

**2026-09-22:** CPU optimization — throttling, reduced polling
frequencies, 5x faster dashboard JS.

---

**One script. Everything runs together, stops together, and stays under 80% CPU.** 🚀
