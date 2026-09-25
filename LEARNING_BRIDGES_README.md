# ETH Ladder Learning Bridges

Three bridges provide access to bot learning data for different LLM environments: **Codex**, **Claude CLI**, and **OMLX**.

## 🌉 Bridge Overview

| Bridge | Location | Use | Access |
|--------|----------|-----|--------|
| **Codex MCP** | `Codex/.integrations/smart-model-router/ethladder_bot_mcp.py` | Structured queries via MCP | Claude in Codex/IDE |
| **Claude CLI** | `ethladder/ethladder_cli_bridge.py` | Command-line queries | Local terminal |
| **OMLX** | `Codex/.integrations/smart-model-router/ethladder_omlx_bridge.py` | Local model queries | OMLX subprocess |

## 🔌 Codex MCP Bridge

### Setup
Already configured in smart-model-router.

### Usage in Codex
```
@ask "get_ethladder_state"        # Current positions and balance
@ask "get_omlx_learning"           # Pattern analysis and calibration
@ask "get_ml_training"             # Model accuracy and trades
@ask "get_walk_forward_results"    # Backtest validation
@ask "get_trading_performance"     # Complete snapshot
```

### Example Query
```
Analyze ETH Ladder's current bounce probability and compare it to 
the moving average of the last 100 trades. What patterns are strongest?

[Uses get_omlx_learning + get_ml_training internally]
```

## 💻 Claude CLI Bridge

### Setup
```bash
cd ~/Documents/ethladder
chmod +x ethladder_cli_bridge.py
```

### Usage
```bash
# Get current state
./ethladder_cli_bridge.py state

# Get OMLX learning data
./ethladder_cli_bridge.py omlx

# Get ML training status
./ethladder_cli_bridge.py ml

# Get backtest results
./ethladder_cli_bridge.py backtest

# Get complete performance snapshot
./ethladder_cli_bridge.py performance

# Watch live updates (refresh every 5 seconds)
./ethladder_cli_bridge.py watch --interval 5

# Export all data to JSON
./ethladder_cli_bridge.py export --output my_export.json
```

### Example Workflow
```bash
# Terminal 1: Watch bot in real-time
./ethladder_cli_bridge.py watch --interval 10

# Terminal 2: Query specific data
./ethladder_cli_bridge.py omlx | jq '.patterns'

# Terminal 3: Export for analysis
./ethladder_cli_bridge.py performance > bot_analysis.json
```

## 🤖 OMLX Local Model Bridge

### Setup
```bash
cd ~/Documents/Codex/.integrations/smart-model-router
chmod +x ethladder_omlx_bridge.py
```

### Usage (Interactive)
```bash
./ethladder_omlx_bridge.py

> What are the current position metrics?
[Outputs bot state analysis]

> Show me OMLX dimension predictiveness
[Outputs OMLX learning analysis]

> How is the ML model performing?
[Outputs training analysis]

> What are the validated parameters?
[Outputs walk-forward analysis]

> quit
```

### Usage (Command Line)
```bash
./ethladder_omlx_bridge.py "Analyze current bounce probability"

./ethladder_omlx_bridge.py "Which patterns are most reliable?"

./ethladder_omlx_bridge.py "Show recent trade outcomes"
```

### Integration with OMLX Server
```bash
# If OMLX is running locally, query it:
curl -X POST http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-Coder-7B",
    "input": "eth ladder: analyze current bounce probability and suggest improvements",
    "max_output_tokens": 2000
  }'
```

## 📊 Data Available

### Bot State
- Open positions (paper & live)
- Trading mode (disabled/trading)
- System metrics (CPU, memory, disk)
- Available balance

### OMLX Learning
- Current bounce probability (%)
- 10-dimension predictiveness scores (100% = perfect)
- Pattern success rates (morning_dip, support_bounce)
- Training events count
- Total calibrated trades

### ML Training
- Total trades analyzed (max 20,000)
- Recent trade outcomes with P&L
- XGBoost accuracy
- RL Q-table state

### Walk-Forward Validation
- Best parameters (leverage, funds %, DCA config)
- Performance metrics (profit factor, win rate, max DD)
- Search progress (% of parameter space tested)
- Next cycle timing

## 🎯 Use Cases

### Pattern Analysis
```bash
# Codex
@ask "Based on OMLX's dimension scores, which features most predict bounces?"

# CLI
./ethladder_cli_bridge.py omlx | jq '.dimensions | sort_by(.) | reverse'

# OMLX
./ethladder_omlx_bridge.py "Which patterns are underutilized?"
```

### Model Improvement
```bash
# Codex
@ask "Review recent trades and suggest XGBoost feature engineering"

# CLI
./ethladder_cli_bridge.py ml | jq '.recent_trades'

# OMLX
./ethladder_omlx_bridge.py "Analyze ML training progress and bottlenecks"
```

### Parameter Validation
```bash
# Codex
@ask "Compare current walk-forward params vs what's live trading"

# CLI
./ethladder_cli_bridge.py backtest

# OMLX
./ethladder_omlx_bridge.py "Are the validated parameters still optimal?"
```

### Real-Time Monitoring
```bash
# CLI watch mode
./ethladder_cli_bridge.py watch --interval 3

# Codex (polling)
@ask "Check bot every 30 seconds and alert on position changes"

# OMLX (scripted)
./ethladder_omlx_bridge.py "Monitor bot health and flag anomalies"
```

## 🔄 Data Update Frequency

| Data | Source | Update |
|------|--------|--------|
| Positions | Live trading/paper bot | Every deal close |
| OMLX metrics | DIP analysis service | Every candle (~1 min) |
| ML training | Continuous trainer | Every 2 minutes |
| Trade memory | Deal outcomes | Every close |
| Walk-forward results | Optimizer | Every 30 minutes |

## ⚠️ Important Notes

- **Read-only access** — Bridges cannot modify bot state
- **No API keys exposed** — Credentials stay in Keychain
- **Real money** — Live trading uses actual Bybit balance
- **14-day backtest** — Validation window is fixed (safety)
- **20k trade cap** — Memory is capped and deduplicated

## 🚀 Quick Start

### For Codex Users
1. Bridges already installed ✓
2. Query directly: `@ask "get_ethladder_state"`
3. Follow up with specific analysis requests

### For CLI Users
1. Run: `./ethladder_cli_bridge.py state`
2. Export data: `./ethladder_cli_bridge.py export`
3. Watch live: `./ethladder_cli_bridge.py watch`

### For OMLX Users
1. Run: `./ethladder_omlx_bridge.py`
2. Type queries interactively
3. Or pass queries as arguments

## 📞 Support

- **Codex MCP issues:** Check `smart-model-router` logs
- **CLI issues:** Run `python ethladder_cli_bridge.py --help`
- **OMLX issues:** Verify OMLX server is running (if using remote mode)
- **Data issues:** Check file existence in `~/Documents/ethladder/`

## 🔗 See Also

- `FINE_TUNING_GUIDE.md` — Detailed fine-tuning and learning guide
- `CLAUDE.md` — Bot architecture and constraints
- `TESTING_POLICY.md` — Safety and validation rules
