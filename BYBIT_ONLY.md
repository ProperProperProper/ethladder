# 🔴 CRITICAL: THIS IS A BYBIT BOT - LOCKED TO BYBIT + REAL DATA ONLY

## ABSOLUTE REQUIREMENT

This system is **BYBIT-ONLY** and uses **REAL MARKET DATA ONLY**.

**NEVER:**
- ❌ Use Binance, Coinbase, Kraken, or any other exchange
- ❌ Use synthetic/generated data
- ❌ Mix data from multiple exchanges
- ❌ Use paper trading prices that don't match BYBIT

**ALWAYS:**
- ✅ Fetch data from BYBIT API (via `fetch_klines()`)
- ✅ Test on real market conditions
- ✅ Use BYBIT's real account balance
- ✅ Trade on BYBIT only

## Why This Matters

1. **Backtests are meaningless if wrong exchange** - BYBIT prices ≠ Binance prices
2. **Learnings are wrong if trained on synthetic** - ML models won't work in real trading
3. **Risk calculations are wrong** - Slippage, fees differ by exchange
4. **Results cannot be trusted** - If you deviate, results are garbage

## Enforcement Points

### Code Level
As of the 2026-09-23 consolidation, everything below runs as a
background task inside `run_everything.py` (see `CLAUDE.md`) rather than
as a standalone script — the enforcement itself hasn't moved:
- `forward_tester` task (was `run_forward_omlx_tester.py`): asserts
  BYBIT data or crashes
- `optimizer` task (was `scripts/continuous_optimizer.py`): uses
  `fetch_klines()` (BYBIT only)
- `symbot_python/ml/continuous_trainer.py`: trains on BYBIT trades only

### Data Sources
- **Forward Tester task**: `fetch_klines()` → BYBIT
- **Optimizer task**: `fetch_klines()` → BYBIT
- **ML Training**: Trade memory from BYBIT trades
- **OMLX Models**: Trained on BYBIT data

### Memory Enforcement
- `memory/real_data_requirement.md`: Absolute BYBIT-only requirement
- This file (`BYBIT_ONLY.md`): Cannot be ignored
- Git history: Every deviation is documented

## What Happens If You Deviate

```
System crashes with: ❌ FATAL: Only ETHUSDT allowed
System crashes with: ❌ FATAL: Cannot continue without BYBIT data
System crashes with: ❌ FATAL: System requires BYBIT for accuracy
```

**The system will NOT run with non-BYBIT data.**

## Review Checklist

Before ANY code change:
- [ ] Does this use BYBIT? (check `fetch_klines()`)
- [ ] Does this use REAL data? (no synthetic fallback)
- [ ] Does this reference other exchanges? (grep for binance, coinbase, etc)
- [ ] Are assertions in place to prevent deviation?

## For Future Claude Sessions

**THIS IS NON-NEGOTIABLE.**

The user has explicitly locked this system to BYBIT + real data.
The code enforces it with assertions.
Memory documents it as absolute.
You MUST NOT deviate.

If I try to use Binance again, the code will crash and refuse to run.
This is by design. It's the right design.

---

**SIGNED: LOCKED 2026-09-22 BY USER DIRECTIVE**
**STATUS: PERMANENT - NO EXCEPTIONS**
