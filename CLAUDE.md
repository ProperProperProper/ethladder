# ETH Ladder - Claude Code Guidelines

This document describes the project architecture, constraints, and guidelines for Claude-assisted development.

## 🎯 Project Overview

**ETH Ladder** is a production trading bot with autonomous ML learning for Ethereum perpetual futures (9-11x leverage on Bybit). It runs 24/7 as **one script** (`run_everything.py`) with:
- Continuous self-improvement via OMLX system
- Real-time dashboard with unified metrics
- SQLite persistence of all data
- Auto-restart on crash (launchd on macOS)

**Current Status:** Production-ready, learning from live forward tests.

## 🔒 Non-Negotiable Constraints

These are LOCKED and should never be changed without explicit user approval:

### Trading Rules
- **Leverage:** Always 9-11x (locked in `symbot_python/strategy/leverage_policy.py`)
- **Instrument:** ETHUSDT only (real trades, never testnet)
- **Account:** Real available balance only (read-only keychain lookup)
- **Backtest Period:** Exactly 14 days strict cap (never longer), always walk-forward validated
- **Risk Limits:** Non-negotiable safety gates — a winner that liquidates even once on fresh data or out-of-sample is demoted/never promoted

### System Requirements
- **Language:** Python 3.12+ only
- **Exchange:** Bybit mainnet only (never testnet)
- **Database:** SQLite only, never CSV, never an external DB
- **Dashboard Port:** 8731 (fixed)
- **CPU Target:** <80% combined usage (critical for system stability)
- **One process:** everything runs inside `run_everything.py` — never reintroduce a separate script that runs standalone alongside it

## 📁 Architecture — Everything Is `run_everything.py`

**There is one entry point.** Web server, dashboard, paper trading bot,
forward tester, continuous optimizer, ML trainer, data exporters, SQLite
saver, resource monitor, data pruner, and log watcher are all asyncio
tasks inside one FastAPI/uvicorn process. There are no other scripts to
run — the old fragmented scripts (`dashboard_server.py`,
`supervise_system.sh`, `continuous_data_exporter.py`,
`continuous_sqlite_saver.py`, `data_persistence.py`,
`export_trading_data.py`, standalone `continuous_optimizer.py` /
`run_forward_omlx_tester.py` / `log_watcher.py`, `data_pruner.py`,
`scheduled_pruner.py`, `export_system_metrics.py`,
`export_live_metrics.py`, and every `DEPLOY*.sh`/`RUN_*.sh` wrapper) were
deleted, not kept for reference — their logic was inlined into
`run_everything.py` directly. If you're tempted to add a new standalone
script for a new piece of functionality, add a background task to
`run_everything.py` instead.

What's imported normally, not inlined: everything under `symbot_python/`
(the real strategy/exchange/ML library code — `dca_bot.py`, `backtest.py`,
`optimize.py`, `forward_omlx_tester.py`, the OMLX modules, etc.). Only the
former *entry-point glue* — the argparse/while-True/signal-handling
wrapper each old top-level script had — was folded into
`run_everything.py`.

### Two structural things that had to be handled deliberately

- **CPU-bound work never runs directly on the shared event loop.** The
  optimizer's walk-forward search (`run_optimizer_cycle`/
  `run_optimizer_interval`) and the forward tester's parameter-combination
  testing are both synchronous, CPU-bound Python with no internal
  `await` points — each runs via `asyncio.to_thread()` wrapping its own
  private `asyncio.run()` call, isolated on a worker thread. Without
  this, uvicorn's own ASGI startup handshake never completes and the
  port never opens (observed directly while building this).
- **Only one FileHandler is attached to the root logger** —
  `logs/run_everything.log`. Any extra FileHandler an imported module's
  own `configure_logging()` call might add is stripped at startup. This
  matters because the log watcher's own alert-about-an-error would
  otherwise get written back into the exact file it's tailing and
  re-trigger itself forever (observed directly: hit 110MB/minute while
  this was being built).
- **The FastAPI app's lifespan is swapped in `main()`, not at module
  import time.** `run_everything.py` is also imported by the test suite
  for its reusable functions (`SEARCH_GRID`, `run_optimizer_cycle`,
  `classify_line`, ...); some other tests use
  `with TestClient(app_module.app)`, which DOES trigger the ASGI
  lifespan. Swapping the lifespan at import time would make an unrelated
  unit test accidentally start all 10 real background tasks.

Machine state: as of 2026-09-24 (later same day), **`com.ethladder.unified`
is installed and loaded** at `~/Library/LaunchAgents/com.ethladder.unified.plist`
— auto-starts on login, auto-restarts on crash (`KeepAlive`/`RunAtLoad`
both true). This reverses an earlier same-day change that had removed
auto-start entirely; it was reinstated on request. `run_everything.py`
runs via the project's `.venv/bin/python`, never the system `python3`
(uvicorn/fastapi/etc. are only installed in `.venv`). This is the only
launchd job for this project — never load any other plist alongside it.
To make code changes take effect: `launchctl unload ...plist && launchctl
load ...plist` (see README.md's "Quick Start").

Also as of 2026-09-24: `log_watcher_loop`'s native macOS desktop
notifications were removed on request (was firing for every ERROR/
CRITICAL log line, including a 2332-in-90-seconds spam episode from
routine backtest activity). Alerts are still fully recorded —
`logs/watcher_alerts.log` and `data/watcher_alerts.db` — just silent on
the desktop. Do not re-add `notify_macos()`/`osascript` calls without
being asked.

### Core Library Modules (`symbot_python/`)

**Trading Engine** (`strategy/`)
- `dca_bot.py` - Per-deal async engine with tick loop
- `backtest.py` - Historical testing with liquidation modeling
- `optimize.py` - Walk-forward parameter search (`walk_forward`, `walk_forward_fixed`)
- `forward_omlx_tester.py` - Simulated trading / OMLX learning data generator
- `optimization_store.py` - SQLite param library (winners, retest/walk-forward records)

**OMLX System** (`exchange/`)
- `dip_analysis_service.py` - Real-time drawdown detection
- `omlx_bounce_analyzer.py` - 10-dimension bounce probability
- `omlx_drawdown_advisor.py` - Safety order decision making
- `dip_calibration_engine.py` - Auto-learns from outcomes; the shared
  `DipCalibrationEngine` singleton both live/paper trading (`dca_bot.py`)
  and the forward-tester (`forward_omlx_tester.py`) record real
  decisions/outcomes into, tagged `DipTradeRecord.source`
  (`"live_paper"`/`"walk_forward"`)
- `inbox_pattern_extractor.py` - Reviews the SEPARATE, external
  training-inbox pipeline's approved Q&A drafts for genuine DCA-strategy
  ideas (never auto-applied — PENDING candidates reviewed via the
  dashboard, `symbot_python/api/inbox_candidates.py`)
- `keychain.py` - Read-only real balance lookup (never places orders)

**ML Training** (`ml/`)
- `continuous_trainer.py` - `ContinuousMLTrainer`, retrains every 2 min
- `trade_memory.py` - `TradeMemory`, capped at 20,000 trades, atomic writes, dedup by report filename
- `outcome_predictor.py` - XGBoost probability predictor
- `rl_decision_optimizer.py` - Q-Learning action selector
- `walk_forward_trainer.py` - Learns from forward tests

**API / Web** (`api/`)
- `app.py` - The FastAPI app (dashboard routes, `fetch_klines`)
- `paper.py` - Paper trading bot manager, `WINNER_PARAM_FIELDS`

**System** (`system/`)
- `cpu_limiter.py` - `get_cpu_limiter()`, throttling thresholds
- `resource_monitor.py` - `SystemResourceMonitor`

### Data Flow

```
Bybit Market Data
    ↓
forward_tester task (30-min cycles, real backtest.py engine)  Paper Trading Bot (in-process, symbot_python/api/paper.py)
    ↓ forward_test_accuracy_*.json                                ↓ live_paper_trades.jsonl (dca_bot.py's _handle_sell)
    └──────────────────────┬─────────────────────────────────────┘
                            ↓
              Trade Memory (trade_memory.json — capped, deduped, BOTH sources)
                            ↓
                  ml_trainer task (every 2 min) — trains outcome_predictor.py/rl_decision_optimizer.py
                            ↓
                  OMLX System (live dip detection, omlx_ml_advisor.py blends model output)

Separately, the SAME shared DipCalibrationEngine (dip_calibration_engine.py)
records every real decision/outcome from BOTH forward_tester and live/paper
trading directly (not file-mediated) — this is the continuous, real-time
pattern/dimension-weight calibration loop, distinct from the offline
XGBoost/RL retraining above.

Inbox candidate patterns (separate, optional): inbox_pattern_extractor task
(hourly) → omlx_candidate_patterns.json → dashboard promote/reject (never
auto-applied to any of the above)

Dashboard & Analytics (omlx_metrics_exporter, data_exporter, sqlite_saver tasks)
```

## ⚡ Key Performance Characteristics

### CPU Management
- **Dashboard refresh:** 15 seconds (browser-side)
- **Data/OMLX-metrics exporter interval:** 30 seconds (CPU-throttled)
- **SQLite saver interval:** 45 seconds (CPU-throttled)
- **Optimizer cycles:** Wait for CPU < 70% before starting a heavy backtest
- **Combined target:** Always < 80% CPU

### Database
- **Tables:** 7 (`omlx_metrics`, `system_metrics`, `paper_trades`, `backtest_results`, `patterns`, `positions`, `daily_summary`)
- **Retention:** 30 days for metrics (7 days for patterns), pruned daily by the `pruner` task
- **Auto-cleanup:** Aggressive cleanup + `VACUUM` triggers if free disk < 5GB
- **Never CSV** — SQLite only, for anything that isn't a one-off JSON snapshot

### Learning Loop
1. `forward_tester` task runs 30-min cycles
2. Trade outcomes accumulated in `trade_memory.json` (capped at 20,000, deduped by report filename)
3. `ml_trainer` task retrains models every 2 minutes
4. OMLX system gets improved bounce/safety predictions
5. Dashboard shows real-time accuracy trends via `omlx_metrics_exporter`

## 🛠️ Development Guidelines

### When Modifying Core Trading Code
- **Never** change leverage bounds without explicit approval
- **Always** test on 14-day backtest windows (not synthetic), always walk-forward
- **Always** verify liquidation modeling / safety gates still hold
- **Never** promote/deploy anything that liquidated even once on out-of-sample data

### When Adding New Continuous Functionality
- **Add it as a background task inside `run_everything.py`**, not a new script
- If it's CPU-bound and synchronous, wrap it in `asyncio.to_thread()` (see `_run_optimizer_cycle_sync` for the pattern)
- Register the task in `unified_lifespan`'s `background` dict
- Consider whether it needs CPU-limiter gating (`cpu_limiter.check_and_throttle(...)`)

### When Adding Features
- **Add corresponding tests** in `tests/` (import reusable functions from `run_everything` directly, same pattern as `test_search_grid_coverage.py`)
- **Update documentation** (README.md, this file)
- **Consider CPU impact** before adding new polling loops

### Code Style
- Use Python 3.12+ type hints
- Keep functions focused
- Add docstrings/comments for the *why*, not the *what* — non-obvious constraints, past incidents, workarounds
- Avoid unnecessary dependencies

## 📊 Key Files to Know

### Configuration
- `symbot_python/strategy/leverage_policy.py` - Leverage constraints
- `symbot_python/strategy/backtest.py` - Backtest constants (`FIXED_TAKE_PROFIT_PERCENT`)
- `symbot_python/signals/candles.py` - `DEFAULT_BACKTEST_DAYS`, `bars_for_days`
- `symbot_python/logging_setup.py` - `configure_logging()`

### Data Files (all at repo root, written by `run_everything.py`'s tasks)
- `trade_memory.json` - All trades, capped at 20,000, atomic writes. Fed
  by BOTH forward-tester reports (`forward_test_accuracy_*.json`) and
  `live_paper_trades.jsonl` (real live/paper deal closes — see
  `symbot_python/ml/trade_memory.py`'s `append_live_paper_trade`/
  `load_live_paper_trades`)
- `live_paper_trades.jsonl` - Append-only log of real live/paper deal
  closes in the same feature schema forward-test reports use, written
  by `dca_bot.py`'s `_handle_sell`. `TradeMemory.live_paper_offset`
  tracks how much of it has been folded into `trade_memory.json`
- `dip_calibration_state.json` - `DipCalibrationEngine`'s persisted
  state: pattern/dimension calibration, `trades` (capped at 5,000
  individual `DipTradeRecord`s, tagged `source: "walk_forward"` or
  `"live_paper"`), `training_events` (capped at 2,000 — one entry per
  real recalibration, what the dashboard's Training Events Timeline
  reads)
- `ml_trainer_status.json` - `ContinuousMLTrainer.get_status()`,
  persisted every check-interval (fixes the dashboard's previously-dead
  "Training Cycles" tile — see `run_everything.py`'s
  `_get_ml_trainer_status()`)
- `omlx_metrics_live.json` - Current OMLX state (dashboard reads this)
- `system_metrics.json` - Current CPU/Memory/Disk/Temp
- `ethladder_analytics.db` - SQLite database
- `omlx_rl_model.json` - Q-table for RL optimizer
- `data/optimizer_status.json` - Live optimizer progress
- `data/watcher_alerts.db` - SQLite alert log (critical issues only)

### Logs
- `logs/run_everything.log` - THE log file — everything writes here (grows
  fast: OMLX dip-analysis logging during forward-test replay alone has
  been observed north of 50 MB/minute — no rotation/truncation exists
  yet, worth keeping an eye on disk usage)
- `logs/optimizer_activity.log` - The optimizer's own `[optimizer]`/
  `[<interval>]`-tagged lines only, ALSO written to run_everything.log
  (purely additive — see run_everything.py's `optimizer_log`). Exists
  because the optimizer's lines are genuinely sparse (one every few
  minutes) relative to this process's total log volume — reading them
  from the combined log by filtering a bounded tail was tried first and
  wasn't reliable at that volume. `/console`'s "Optimizer Activity" tab
  reads this file directly
- `logs/watcher_alerts.log` - Plain-text fallback for critical alerts

## 🔍 Testing Guidelines

### Before Committing
1. **Syntax check:** `python3 -m py_compile run_everything.py` (compile-only, works with any Python)
2. **Import check:** `.venv/bin/python -c "import run_everything"` (must be `.venv` — uvicorn/fastapi/etc. aren't installed for the system `python3`)
3. **Run the test suite:** `.venv/bin/python -m pytest tests/ -q`
4. **Git diff review:** `git diff --staged`

Full suite is 415/415 passing as of 2026-09-23. Any failure now is new —
don't assume it's pre-existing without checking (`git stash` your
changes and re-run to confirm). See "Recent Significant Changes" below
for the four distinct bugs that were fixed to get here.

### For Trading Changes
1. **Backtest on 14 days real data** (never synthetic)
2. **Verify liquidation never triggers** (or verify intended)
3. **Check profit factor** (should be > 1.0)
4. **Review win rate** (should be > 50%)

### For System Changes
1. **Check CPU impact** - Does it reduce overhead?
2. **Run it live** - `.venv/bin/python run_everything.py`, wait for `PORT OPENED`, curl `/console`
3. **Monitor for a minute or two** - `grep -c "ERROR\|CRITICAL" logs/run_everything.log`
4. **Test graceful shutdown** - `kill -TERM <pid>`, confirm the port releases

## 🚀 Deployment Process

1. **Commit changes** with clear message
2. **Verify compilation** - No syntax errors
3. **Run it live for a minute** - confirm all 10 background tasks start, zero errors
4. **Run the test suite** - confirm no new failures beyond the known pre-existing ones
5. **Update documentation** - README.md, this file
6. **Push to main** - Always merge to main for production

**No long-running branches** - Keep work on main or short-lived feature branches.

## 📞 Emergency Procedures

### System Won't Start
```bash
# First unload the launchd job or it'll relaunch and fight for port 8731:
launchctl unload ~/Library/LaunchAgents/com.ethladder.unified.plist
.venv/bin/python run_everything.py   # foreground — read the traceback directly
# When done debugging, restore the service:
launchctl load ~/Library/LaunchAgents/com.ethladder.unified.plist
```

### High CPU Usage
```bash
top
python3 -c "from symbot_python.system.cpu_limiter import get_cpu_limiter; l=get_cpu_limiter(); print(l.get_status_summary())"
tail -f logs/run_everything.log
```

### Database Too Large
The `pruner` task runs once daily automatically. To force it immediately:
```bash
python3 -c "
import run_everything as re
re._prune_once()
"
```

### Dashboard Not Loading
```bash
curl http://127.0.0.1:8731/console
lsof -i :8731
launchctl list | grep ethladder

# Restart via launchd (see README.md's "Quick Start")
launchctl unload ~/Library/LaunchAgents/com.ethladder.unified.plist
launchctl load ~/Library/LaunchAgents/com.ethladder.unified.plist
```

## 🔄 Recent Significant Changes

**2026-09-24 (latest):** Auto-start reinstated on request — reverses the
"remove auto-start" change two entries below. Diagnosed why the web
server wasn't running: nothing had restarted it after the launchd job
was unloaded, and separately, running it with the bare system `python3`
fails (`ModuleNotFoundError: No module named 'uvicorn'`) — dependencies
only exist in `.venv`, which the plist already correctly pointed at.
Installed `com.ethladder.unified.plist` to `~/Library/LaunchAgents/`,
loaded it, verified: process running, port 8731 listening, `/console`
returns 200, no fresh errors. Desktop notifications remain removed —
that part of the earlier change was not reversed. Also fixed a stale
`curl http://127.0.0.1:8731/api/status` troubleshooting command in this
file and README.md — that endpoint doesn't exist; real routes are
`/console`, `/paper`, `/winners`, `/api/optimizer-status`, `/api/alerts`,
`/api/logs`.

**2026-09-24 (later):** Git hygiene + doc cleanup
- Untracked and deleted from disk 88 runtime/scratch files that an
  early, over-broad `git add -A` had committed (`trade_memory.json` had
  reached 1.3GB in history) — `.gitignore` updated so they can't be
  re-added
- GitHub rejected the push over the 1.3GB blob in history even after
  untracking it going forward (files still in old commits' trees); used
  `git filter-repo` to strip these paths out of history entirely (all
  local commits, none pushed yet at the time) and force-pushed the
  cleaned history to `origin/main`
- Deleted 23 stale root-level markdown docs that described the old
  fragmented multi-script architecture (`run_forward_omlx_tester.py`,
  `scripts/continuous_optimizer.py`, `DEPLOY.sh`, etc. — all long since
  deleted) or were pre-implementation design/planning docs superseded by
  the actual OMLX code and this file. Kept `TESTING_POLICY.md` and
  `BYBIT_ONLY.md` (both are non-negotiable policy docs referenced from
  live code/comments) and fixed their stale script references to point
  at `run_everything.py`'s background tasks instead
- Verified no secrets/credentials/PII in tracked files or git history —
  this project never stores exchange API keys in code (see
  `symbot_python/exchange/keychain.py`'s docstring: macOS Keychain only,
  read-only, credentials never logged/persisted)

**2026-09-24:** Several fixes to OMLX training data, position sizing, and operational behavior
- OMLX's ML training data was mostly hardcoded stubs (6 of 11
  `outcome_predictor.py` features never varied; `current_loss_pct` was
  always 0) — now real, pulled from `DrawdownDecision.context`/
  `.dimension_scores` (added to carry `DipContext`/`BounceAnalysis`
  through instead of discarding them)
- OMLX Learning dashboard tab showed accuracy/precision/recall/F1/
  patterns all as 0 — `forward_omlx_tester.py` was directly mutating
  win/loss counters instead of calling `record_trade_decision()`/
  `record_trade_outcome()` (the only path that computes the confusion
  matrix); `run_everything.py`'s dashboard export was also reading keys
  (`total_trades`, `pattern_success_rates`) that never existed
- Paper trading falls back to a simulated $200 USDT starting balance
  when the real available balance is below $1 (was showing "0.0000" —
  turned out to be $0.00003903 dust, not literally 0, so a naive
  `<= 0` check never fired)
- New `funds_utilization_policy.py`: hard floor of 25% on
  `funds_utilization_percent`, same pattern/enforcement point as
  `leverage_policy.py` — a deal sized below this can't produce a
  meaningful profit even off a fully-successful ladder
- Fixed 2332-in-90-seconds CRITICAL desktop-alert spam: `forward_omlx_tester.py`'s
  own `DipAnalysisService` (not the live/paper singleton) was logging
  routine backtest dips as `CRITICAL`; added `emit_critical_alerts` flag,
  `False` for the forward tester
- Removed `log_watcher`'s native macOS desktop notifications and the
  project's launchd auto-restart/auto-start entirely, on request — see
  "Architecture" above and README.md's "No Auto-Start" section

**2026-09-23:** Fixed all 31 pre-existing test failures — 415/415 passing. Four distinct bugs, not one:
1. `/paper` route regression — an earlier session (`812e760`) replaced
   the real interactive control page (stop/cancel/panic, param updates,
   equity chart) with a static dashboard copy, leaving debug `print()`s
   in place. Reverted `paper_page` to render `paper.html` again.
2. The dashboard's SPA catch-all route was also swallowing genuinely
   undefined `/api/*` paths, always returning 200 instead of 404 — made
   it impossible to verify a removed endpoint was actually gone. Now
   excludes `path.startswith("api/")`.
3. `PaperExchangeClient.place_market_order()` filled at the bid/ask
   **midpoint** instead of crossing the spread (comment: "to eliminate
   spread noise when debugging" — never reverted). Buy now fills at ask,
   Sell at bid; `opposite_price` (feeds `reverse_paper.py`'s mirror) is
   the genuinely opposite quote. Was making paper P&L unrealistically
   optimistic vs. live (zero spread cost).
4. `_try_start_deal` used `bot = replace(bot, side=direction)` —
   `dataclasses.replace()` returns a new object, silently detaching the
   local `bot` from `self.bots[bot_id]` whenever OMLX returned a
   concrete direction (the normal case). Every `bot.deal_count += 1`
   after that mutated an orphaned copy — `deal_max` caps never actually
   triggered in production. Fixed by mutating `bot.side` in place
   instead (`BotConfig` is already a plain mutable dataclass).

**2026-09-23:** Merged everything into one script (`run_everything.py`), deleted the old fragmented scripts entirely (not kept for rollback)
- One process, one PID, 10 background tasks — see "Architecture" above
- Found and fixed the likely actual cause of an earlier machine crash:
  `trade_memory.json` had grown to **5.7GB**. `load_all_forward_test_trades()`
  re-ingested every `forward_test_accuracy_*.json` report from scratch on
  every 2-minute ML training cycle, forever, with no dedup —
  `TradeMemory` now tracks `processed_reports` so each report file is
  only folded in once, `save()` writes atomically (temp file +
  `os.replace()`), and a hard `MAX_TRADES = 20_000` cap makes unbounded
  growth structurally impossible regardless of upstream volume
- Fixed a SQLite `check_same_thread` crash (connection created on the
  event-loop thread, used via `asyncio.to_thread` on a different one
  each call) — now `check_same_thread=False`, safe because access is
  always sequential
- Fixed a log-watcher self-feedback loop that hit 110MB/minute (see
  "Architecture" above)
- Removed module-level `logging.basicConfig()` calls that were silently
  overriding whichever entry point imported them first (this bug no
  longer applies — those modules were inlined and deleted)
- Added `scikit-learn`, `xgboost`, `numpy` to `pyproject.toml` (were
  imported by `symbot_python/ml` but never declared)
- Added two components that were missing from the original merge:
  `omlx_metrics_exporter` task (was `export_live_metrics.py`, the
  dashboard's OMLX Learning tab was silently stale without it) and
  `resource_monitor`/`pruner` tasks
- Removed `com.ethladder.web` / `.optimizer` / `.watcher` / `com.omlx.system`
  launchd jobs and plists — `com.ethladder.unified` is the only one now
- Updated 4 test files (`test_continuous_optimizer_resilience.py`,
  `test_continuous_optimizer_status.py`, `test_log_watcher.py`,
  `test_search_grid_coverage.py`) to import from `run_everything`
  instead of the deleted `scripts.continuous_optimizer`/`scripts.log_watcher`

**2026-09-22:** CPU optimization — throttling module, reduced polling frequencies, 5x faster dashboard JS

**Earlier:** Unified master dashboard, SQLite persistence, ML training (XGBoost + RL), OMLX system with 10-dimension analysis

## ✅ Pre-Commit Checklist

- [ ] Code compiles (`python3 -m py_compile run_everything.py`)
- [ ] `python3 -c "import run_everything"` succeeds
- [ ] `python3 -m pytest tests/ -q` shows no new failures
- [ ] Ran it live, confirmed all 10 tasks start with zero errors
- [ ] Documentation updated
- [ ] CPU impact considered
- [ ] No new top-level script created — new functionality is a task inside `run_everything.py`

---

**Remember:** This is a production trading system. Changes should be conservative and well-tested. When in doubt, ask the user first.
