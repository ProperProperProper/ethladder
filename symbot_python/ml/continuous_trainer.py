"""Continuous ML Trainer - Runs forever, constantly improving models.

Decoupled from forward tester. Runs in parallel, constantly:
- Checks for new forward test data
- Retrains all models
- Updates every few minutes
- Never stops improving
"""

import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path

from symbot_python.ml.walk_forward_trainer import WalkForwardTrainer
from symbot_python.ml.trade_memory import (
    LIVE_PAPER_TRADES_FILE,
    TradeMemory,
    load_all_forward_test_trades,
    load_live_paper_trades,
)

logger = logging.getLogger(__name__)


class ContinuousMLTrainer:
    """Runs forever, continuously retraining on new data."""

    def __init__(self, check_interval_seconds: int = 120, status_file: str = "ml_trainer_status.json"):
        """Initialize continuous trainer.

        Args:
            check_interval_seconds: How often to check for new data (default 2 min)
            status_file: Where get_status() is persisted after every loop
                iteration (see _save_status) — run_everything.py's
                _get_current_omlx_metrics() reads this to populate the
                dashboard's "Training Cycles" tile. That tile previously
                always read a key that was never set anywhere
                (progress.training_cycles), showing "0" forever
                regardless of real activity — this instance's live state
                (training_count, last_training_time) existed only inside
                the asyncio.to_thread() closure run_everything.py's
                ml_trainer_loop() runs it in, never exposed to any file,
                the dashboard, or SQLite.
        """
        self.check_interval = check_interval_seconds
        self.status_file = Path(status_file)
        self.trainer = WalkForwardTrainer()
        self.memory = TradeMemory('trade_memory.json')
        self.last_training_time = None
        self.training_count = 0
        self.last_report_count = 0
        self.last_check_time = None
        self.last_check_found_new_reports = 0

    def check_for_new_data(self) -> int:
        """Check if new forward-test reports OR live/paper trade closes
        exist since the last check.

        Returns:
            Number of new reports/trades since last check (report count
            delta + live/paper trades appended since our saved offset —
            same shape as before, just no longer forward-test-only).
        """
        reports = sorted(Path('.').glob('forward_test_accuracy_*.json'))
        new_reports = len(reports) - self.last_report_count
        live_paper_path = Path(LIVE_PAPER_TRADES_FILE)
        new_live_paper = 0
        if live_paper_path.exists():
            new_live_paper = max(0, live_paper_path.stat().st_size - self.memory.live_paper_offset)
        return new_reports + (1 if new_live_paper > 0 else 0)

    def train_cycle(self):
        """Run one training cycle on all available data."""
        logger.info("\n" + "=" * 80)
        logger.info(f"🔄 CONTINUOUS ML TRAINING CYCLE #{self.training_count + 1}")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        try:
            # Load all trades into memory
            logger.info("\n📥 Loading all forward test data...")
            total_trades, self.memory = load_all_forward_test_trades(
                self.memory,
                limit_reports=None  # ALL reports, not just 10
            )
            live_paper_trades = load_live_paper_trades(self.memory)
            total_trades += live_paper_trades
            if live_paper_trades:
                self.memory.save()  # load_live_paper_trades doesn't save itself — see its own docstring

            if total_trades == 0:
                logger.warning("⚠️  No new trades to train on")
                return False

            # Train all models
            self.trainer.collected_trades = self.memory.get_all_trades()
            logger.info(f"\n🧠 Training on {len(self.trainer.collected_trades)} total trades...")

            self._train_models()

            self.training_count += 1
            self.last_training_time = datetime.now()
            self.last_report_count = len(list(Path('.').glob('forward_test_accuracy_*.json')))

            logger.info(f"\n✅ Training cycle #{self.training_count} complete")
            logger.info(f"   Next check in {self.check_interval}s")
            return True

        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            return False

    def _train_models(self):
        """Train all models in sequence."""
        logger.info("  → XGBoost Outcome Predictor")
        self.trainer._train_outcome_predictor()

        logger.info("  → RL Decision Optimizer")
        self.trainer._train_rl_optimizer()

        logger.info("  → OMLX Pattern Analysis")
        self.trainer._train_omlx_patterns()

    def run_forever(self):
        """Run continuous training loop forever."""
        logger.info("\n" + "╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 78 + "║")
        logger.info("║" + "CONTINUOUS ML TRAINER - RUNNING FOREVER".center(78) + "║")
        logger.info("║" + f"Retraining every {self.check_interval}s on ALL accumulated data".center(78) + "║")
        logger.info("║" + " " * 78 + "║")
        logger.info("╚" + "=" * 78 + "╝")

        cycle = 0
        while True:
            cycle += 1
            try:
                self.last_check_time = datetime.now()
                new_data = self.check_for_new_data()
                self.last_check_found_new_reports = new_data

                if new_data > 0:
                    logger.info(f"\n📊 Found {new_data} new forward test reports")
                    self.train_cycle()
                else:
                    logger.info(f"\n⏳ No new data. Waiting {self.check_interval}s...")

                self._save_status()

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("\n\n🛑 STOPPING CONTINUOUS TRAINER")
                logger.info(f"   Completed {self.training_count} training cycles")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(self.check_interval)

    def _save_status(self) -> None:
        """Persist get_status() so a separate process (run_everything.py's
        omlx_metrics_exporter_loop) can read real training activity —
        this instance otherwise only exists inside ml_trainer_loop's
        asyncio.to_thread() closure. Atomic write, same reasoning as
        TradeMemory.save()."""
        try:
            status = self.get_status()
            status["last_check_time"] = self.last_check_time.isoformat() if self.last_check_time else None
            status["last_check_found_new_reports"] = self.last_check_found_new_reports
            tmp_path = self.status_file.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
            os.replace(tmp_path, self.status_file)
        except Exception as e:
            logger.error(f"Failed to save trainer status: {e}")

    def get_status(self) -> dict:
        """Get trainer status."""
        return {
            'training_cycles': self.training_count,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'total_trades_in_memory': len(self.memory.get_all_trades()),
            'memory_stats': self.memory.get_stats(),
            'check_interval_seconds': self.check_interval
        }


def start_continuous_trainer(check_interval: int = 120):
    """Start the continuous ML trainer."""
    trainer = ContinuousMLTrainer(check_interval_seconds=check_interval)
    trainer.run_forever()


if __name__ == '__main__':
    import logging as log_module
    log_module.basicConfig(
        level=log_module.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run with 2-minute check interval
    start_continuous_trainer(check_interval=120)
