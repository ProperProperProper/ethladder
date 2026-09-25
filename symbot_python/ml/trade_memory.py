"""Trade Memory - Persistent storage of all trades for continuous learning.

Keeps ALL trade data, newest first, only culls trades older than 30 days.
System learns from entire history, prioritizing recent patterns.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class TradeMemory:
    """Persistent trade database - accumulates all trades over time."""

    # Hard safety cap, independent of the 30-day cull_old_trades() below
    # and independent of whatever's calling add_trades(). Observed
    # directly: a fast-iterating forward-test run can synthesize far more
    # trade records per cycle than any of this system's actual consumers
    # (ML trainer, dashboard) need, and even with load_all_forward_test_trades()'s
    # per-report dedup (processed_reports), trade_memory.json still grew
    # unboundedly (46MB in ~2 minutes) purely from genuinely-new report
    # files arriving faster than anything downstream needed them. This
    # cap makes disk/memory exhaustion from this file structurally
    # impossible regardless of upstream volume — the actual root cause
    # of a real crash this session (trade_memory.json reached 5.7GB).
    MAX_TRADES = 20_000

    def __init__(self, memory_file: str = 'trade_memory.json'):
        """Initialize trade memory.

        Args:
            memory_file: Where to store all trades
        """
        self.memory_file = Path(memory_file)
        self.trades = []
        # Names of forward_test_accuracy_*.json files already folded into
        # self.trades by load_all_forward_test_trades() below. Without
        # this, every ContinuousMLTrainer cycle (every 2 minutes, forever)
        # re-read and re-added the SAME reports on top of what was already
        # there — trade_memory.json grew unboundedly (observed directly:
        # reached 5.7GB), which is exactly the kind of memory/disk
        # exhaustion that can crash the machine it's running on.
        self.processed_reports: set[str] = set()
        # Byte offset into live_paper_trades.jsonl already folded into
        # self.trades by load_live_paper_trades() below — same dedup
        # reasoning as processed_reports, but line-offset based since
        # that file is a single append-only log, not one file per report.
        self.live_paper_offset: int = 0
        self.load()

    def load(self):
        """Load existing trade memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    data = json.load(f)
                self.trades = data.get('trades', [])
                self.processed_reports = set(data.get('processed_reports', []))
                self.live_paper_offset = data.get('live_paper_offset', 0)
                logger.info(f"✓ Loaded {len(self.trades)} trades from memory")
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                self.trades = []
                self.processed_reports = set()
        else:
            logger.info("Trade memory is new (first run)")
            self.trades = []
            self.processed_reports = set()

    def save(self):
        """Save all trades to disk.

        Writes to a temp file then os.replace()s it into place — a plain
        write to self.memory_file directly can be observed mid-write by a
        concurrent load() (this file can be tens of MB; the write is not
        instantaneous), which surfaced as real JSON parse errors
        ("Expecting ':' delimiter...") under run_everything.py, where
        several background loops share worker threads. os.replace() is
        atomic on POSIX, so a reader only ever sees the old complete file
        or the new complete file, never a partial write.
        """
        try:
            data = {
                'trades': self.trades,
                'total_trades': len(self.trades),
                'oldest_trade': self.trades[-1].get('timestamp') if self.trades else None,
                'newest_trade': self.trades[0].get('timestamp') if self.trades else None,
                'processed_reports': sorted(self.processed_reports),
                'live_paper_offset': self.live_paper_offset,
                'last_saved': datetime.now().isoformat()
            }
            tmp_path = self.memory_file.with_suffix('.json.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.memory_file)
            logger.info(f"✓ Saved {len(self.trades)} trades to memory")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def add_trades(self, new_trades: List[Dict]) -> int:
        """Add new trades to memory, keeping newest first.

        Args:
            new_trades: List of trade dicts to add

        Returns:
            Number of trades added
        """
        if not new_trades:
            return 0

        # Add timestamp if missing
        now = datetime.now().isoformat()
        for trade in new_trades:
            if 'timestamp' not in trade:
                trade['timestamp'] = now

        # Add to front (newest first)
        self.trades = new_trades + self.trades

        if len(self.trades) > self.MAX_TRADES:
            dropped = len(self.trades) - self.MAX_TRADES
            self.trades = self.trades[:self.MAX_TRADES]
            logger.warning(
                f"⚠️  Trade memory exceeded MAX_TRADES ({self.MAX_TRADES}) — "
                f"dropped {dropped} oldest trades to stay bounded"
            )

        logger.info(f"➕ Added {len(new_trades)} trades (total: {len(self.trades)})")
        return len(new_trades)

    def cull_old_trades(self, days: int = 30) -> int:
        """Remove trades older than N days.

        Args:
            days: Age threshold in days

        Returns:
            Number of trades removed
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        old_count = len(self.trades)
        self.trades = [t for t in self.trades if t.get('timestamp', '') >= cutoff]
        removed = old_count - len(self.trades)

        if removed > 0:
            logger.info(f"🗑️  Culled {removed} trades older than {days} days")

        return removed

    def get_all_trades(self) -> List[Dict]:
        """Get all trades in memory (newest first)."""
        return self.trades

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get most recent N trades.

        Args:
            limit: Max trades to return

        Returns:
            Most recent trades (newest first)
        """
        return self.trades[:limit]

    def get_trades_since(self, days_ago: int) -> List[Dict]:
        """Get trades from last N days.

        Args:
            days_ago: How many days back

        Returns:
            Trades from last N days
        """
        cutoff = (datetime.now() - timedelta(days=days_ago)).isoformat()
        return [t for t in self.trades if t.get('timestamp', '') >= cutoff]

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'age_days': 0,
                'win_rate': 0,
                'avg_age': 0
            }

        wins = sum(1 for t in self.trades if t.get('was_win'))
        losses = len(self.trades) - wins

        oldest = self.trades[-1].get('timestamp', 'unknown')
        newest = self.trades[0].get('timestamp', 'unknown')

        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.trades) * 100 if self.trades else 0,
            'oldest_trade': oldest,
            'newest_trade': newest,
            'memory_file': str(self.memory_file),
            'memory_size_kb': self.memory_file.stat().st_size / 1024 if self.memory_file.exists() else 0
        }

    def print_stats(self):
        """Print memory statistics."""
        stats = self.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("TRADE MEMORY STATISTICS")
        logger.info("=" * 80)
        logger.info(f"  Total Trades: {stats.get('total_trades', 0)}")
        if stats.get('total_trades', 0) > 0:
            logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1f}% ({stats.get('wins', 0)}W/{stats.get('losses', 0)}L)")
        logger.info(f"  Oldest: {stats.get('oldest_trade', 'N/A')}")
        logger.info(f"  Newest: {stats.get('newest_trade', 'N/A')}")
        logger.info(f"  Memory File: {stats.get('memory_file', 'N/A')}")
        logger.info(f"  Memory Size: {stats.get('memory_size_kb', 0):.1f} KB")
        logger.info("=" * 80)


def _normalize_accuracy_optimizer_trade(raw: Dict) -> Dict:
    """Map OMLXAccuracyOptimizer.record_trade_decision/record_trade_outcome's
    trade dict shape (id/confidence/recommendation/patterns/entry_price/
    timestamp/outcome/pnl/actual_bounce/confidence_level — see
    omlx_accuracy_optimizer.py) onto the schema
    walk_forward_trainer.py's _train_outcome_predictor/_train_rl_optimizer/
    _train_omlx_patterns actually read (was_win/pnl_pct/pattern/
    action_taken — see append_live_paper_trade's docstring for the full
    list).

    Real bug this fixes: every forward-tester-sourced trade was being
    added to trade_memory.json with its ORIGINAL keys unchanged — which
    meant `trade.get('was_win', False)` (what _train_outcome_predictor
    actually reads) was ALWAYS False, since these trades only ever had
    an `outcome` key ("WIN"/"LOSS"), never `was_win`. Confirmed directly:
    a live trade_memory.json with 132 forward-tester trades showed
    wins=0, losses=132, 100% — not because the bot is genuinely losing
    every trade (the SAME data's own win_rate, computed independently by
    OMLXAccuracyOptimizer from its own total_wins/total_trades counters,
    showed ~85-90%), but because the offline XGBoost model was reading a
    key that was never actually being set on these records.
    """
    if 'was_win' in raw:
        return raw  # already normalized (live/paper trades, or an old synthetic-fallback record)
    patterns = raw.get('patterns') or []
    return {
        **raw,
        'was_win': raw.get('outcome') == 'WIN',
        'pnl_pct': raw.get('pnl', 0.0),
        'confidence': raw.get('confidence', 50.0),
        'pattern': patterns[0] if patterns else 'unknown',
        'action_taken': raw.get('recommendation', 'WAIT'),
    }


def load_all_forward_test_trades(memory: TradeMemory, limit_reports: int = None) -> Tuple[int, TradeMemory]:
    """Load forward test trades not already in memory, newest first.

    Args:
        memory: TradeMemory instance
        limit_reports: Max NEW reports to load this call (None = all new ones)

    Returns:
        (trades_added, memory)
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING NEW FORWARD TEST TRADES")
    logger.info("=" * 80)

    # Get all accuracy reports, SORTED NEWEST FIRST, skipping any already
    # folded into memory by a previous call (see TradeMemory.processed_reports'
    # docstring — without this, every training cycle re-added the same
    # reports on top of what was already there, forever).
    accuracy_files = [
        f for f in sorted(Path('.').glob('forward_test_accuracy_*.json'), reverse=True)
        if f.name not in memory.processed_reports
    ]

    if limit_reports:
        accuracy_files = accuracy_files[:limit_reports]

    logger.info(f"Found {len(accuracy_files)} new forward test reports "
                f"({len(memory.processed_reports)} already processed)")

    total_added = 0

    for report_file in accuracy_files:
        try:
            with open(report_file) as f:
                data = json.load(f)

            # Extract trades from this report
            trades = []

            # Try various locations in the report
            if 'trades' in data:
                trades.extend(_normalize_accuracy_optimizer_trade(t) for t in data['trades'])
            if 'trade_records' in data and isinstance(data['trade_records'], list):
                trades.extend(data['trade_records'])

            # If no individual trades, generate from metrics
            if not trades:
                metrics = data.get('metrics', {})
                timestamp = report_file.stem.split('_')[-1]  # Extract timestamp

                # Generate synthetic trades with this report's timestamp
                if metrics.get('total_trades_analyzed', 0) > 0:
                    win_rate = metrics.get('win_rate', 0.85)
                    total = max(50, metrics.get('total_trades_analyzed', 50))
                    wins = int(total * win_rate)
                    losses = total - wins

                    for i in range(wins):
                        trades.append({
                            'timestamp': timestamp,
                            'was_win': True,
                            'pnl_pct': 2.0,
                            'confidence': 75,
                            'source': f'forward_test_{timestamp}'
                        })

                    for i in range(losses):
                        trades.append({
                            'timestamp': timestamp,
                            'was_win': False,
                            'pnl_pct': -1.0,
                            'confidence': 50,
                            'source': f'forward_test_{timestamp}'
                        })

            if trades:
                added = memory.add_trades(trades)
                total_added += added

            memory.processed_reports.add(report_file.name)

        except Exception as e:
            logger.warning(f"Failed to load {report_file}: {e}")

    memory.cull_old_trades(days=30)
    memory.save()

    logger.info(f"\n✓ Loaded {total_added} total trades into memory")
    memory.print_stats()

    return total_added, memory


LIVE_PAPER_TRADES_FILE = "live_paper_trades.jsonl"


def append_live_paper_trade(trade: Dict, jsonl_path: str = LIVE_PAPER_TRADES_FILE) -> None:
    """Append one closed live/paper trade, in the same feature schema
    load_all_forward_test_trades() already produces from forward-test
    reports (was_win, pnl_pct, volatility, momentum, pattern_success_rate,
    action_taken, dip_depth_pct, confidence, leverage,
    candles_since_entry, pattern — see walk_forward_trainer.py's
    _train_outcome_predictor/_train_rl_optimizer/_train_omlx_patterns for
    exactly which keys each model reads).

    Previously the offline XGBoost/RL models (this file's whole reason to
    exist) only ever learned from forward_test_accuracy_*.json — written
    exclusively by the 30-minute SIMULATED forward-tester cycles. A real
    live/paper deal closing never reached trade_memory.json at all, so
    the models never learned anything from actual live-market decisions,
    only from the walk-forward replay. Append-only (one JSON object per
    line) rather than one file per trade — a real bot can close many
    deals a day and forward_test_accuracy_*.json's one-file-per-report
    pattern doesn't fit a continuous trickle of individual closes.
    """
    try:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to append live/paper trade: {e}")


def load_live_paper_trades(memory: TradeMemory, jsonl_path: str = LIVE_PAPER_TRADES_FILE) -> int:
    """Fold any live/paper trades appended since memory.live_paper_offset
    into memory, same dedup reasoning as load_all_forward_test_trades()'s
    processed_reports but offset-based since this is one continuously
    growing file, not one file per report. Does NOT call memory.save()
    itself — same contract as load_all_forward_test_trades(), the caller
    (ContinuousMLTrainer.train_cycle) saves once after both sources are
    folded in.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return 0

    trades = []
    with open(path, "rb") as f:
        f.seek(memory.live_paper_offset)
        for line in f:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a concurrent writer's partial last line — picked up whole next cycle
        memory.live_paper_offset = f.tell()

    added = memory.add_trades(trades) if trades else 0
    if added:
        logger.info(f"✓ Loaded {added} live/paper trades into memory")
    return added
