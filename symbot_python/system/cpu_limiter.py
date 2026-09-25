"""CPU limiter - dynamically throttle processes to keep system under 80% CPU."""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

CPU_THRESHOLD = 80.0  # Max CPU usage %
CPU_WARNING = 75.0   # Start throttling at this level
SAMPLING_INTERVAL = 2.0  # Check CPU every 2 seconds
MIN_SLEEP_MS = 10  # Minimum throttle sleep in milliseconds


class CPULimiter:
    """Dynamically throttle heavy processes based on CPU usage."""

    def __init__(self, threshold: float = CPU_THRESHOLD, warning_level: float = CPU_WARNING):
        """Initialize CPU limiter.

        Args:
            threshold: Hard CPU limit (%)
            warning_level: Start throttling at this level (%)
        """
        self.threshold = threshold
        self.warning_level = warning_level
        self.last_check_time = 0
        self.last_cpu_percent = 0
        self.throttle_factor = 1.0  # 1.0 = no throttle, 0.5 = 50% throttle
        self.stats = {
            'throttle_events': 0,
            'last_throttle_time': None,
            'total_throttle_ms': 0,
            'peak_cpu': 0,
        }

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percent."""
        try:
            result = subprocess.run(['top', '-l', '2', '-n', '0'],
                                  capture_output=True, text=True, timeout=5)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CPU usage:' in line:
                    # Parse: "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'user' in part and i > 0:
                            try:
                                cpu_percent = float(parts[i-1].rstrip('%'))
                                return cpu_percent
                            except:
                                pass
                    break
        except:
            pass
        return self.last_cpu_percent

    def check_and_throttle(self, work_name: str = "background_work") -> bool:
        """Check CPU and throttle if needed.

        Call this before doing CPU-heavy work. Returns True if work should proceed normally.
        Returns False if work should be skipped/deferred this cycle.

        Args:
            work_name: Name of the work being throttled (for logging)

        Returns:
            True if work should proceed, False if it should be skipped
        """
        now = time.time()

        # Only check CPU every SAMPLING_INTERVAL seconds to avoid overhead
        if now - self.last_check_time < SAMPLING_INTERVAL:
            return self.should_proceed()

        self.last_check_time = now
        cpu_percent = self.get_cpu_percent()
        self.last_cpu_percent = cpu_percent

        if cpu_percent > self.stats['peak_cpu']:
            self.stats['peak_cpu'] = cpu_percent

        if cpu_percent > self.threshold:
            # Hard limit exceeded - skip this cycle
            logger.warning(f"CPU at {cpu_percent:.1f}% (threshold {self.threshold}%) - {work_name} deferred")
            self.stats['throttle_events'] += 1
            self.stats['last_throttle_time'] = datetime.now().isoformat()
            return False

        elif cpu_percent > self.warning_level:
            # Warning level - add exponential backoff
            excess = cpu_percent - self.warning_level
            backoff_percent = min(excess * 2, 90)  # Scale 0-90% based on excess
            sleep_ms = max(MIN_SLEEP_MS, int(100 * backoff_percent / 100))

            logger.info(f"CPU at {cpu_percent:.1f}% - throttling {work_name} ({sleep_ms}ms)")
            self.stats['throttle_events'] += 1
            self.stats['last_throttle_time'] = datetime.now().isoformat()
            self.stats['total_throttle_ms'] += sleep_ms

            time.sleep(sleep_ms / 1000.0)  # Convert to seconds
            return True

        return True

    def should_proceed(self) -> bool:
        """Quick check if work should proceed (without doing CPU measurement)."""
        if self.last_cpu_percent > self.threshold:
            return False
        return True

    def wait_until_safe(self, target_cpu: float = 70.0, timeout_sec: float = 60.0) -> bool:
        """Block until CPU drops below target.

        Useful for critical sections that must complete but can wait for CPU to drop.

        Args:
            target_cpu: Target CPU usage to wait for (%)
            timeout_sec: Give up after this many seconds

        Returns:
            True if CPU dropped to target, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout_sec:
            cpu_percent = self.get_cpu_percent()
            if cpu_percent < target_cpu:
                logger.info(f"CPU dropped to {cpu_percent:.1f}% after {time.time() - start:.1f}s")
                return True
            logger.info(f"Waiting for CPU to drop from {cpu_percent:.1f}% to {target_cpu:.1f}%...")
            time.sleep(2)

        logger.warning(f"Timeout waiting for CPU to drop to {target_cpu}%")
        return False

    def export_stats(self, output_file: str = 'cpu_limiter_stats.json') -> bool:
        """Export throttling statistics."""
        try:
            data = {
                'threshold': self.threshold,
                'warning_level': self.warning_level,
                'current_cpu': self.last_cpu_percent,
                'stats': self.stats,
                'last_updated': datetime.now().isoformat()
            }
            tmp_path = f"{output_file}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, output_file)
            return True
        except Exception as e:
            logger.error(f"Failed to export stats: {e}")
            return False

    def get_status_summary(self) -> dict:
        """Get current throttle status."""
        return {
            'cpu_percent': self.last_cpu_percent,
            'threshold': self.threshold,
            'warning_level': self.warning_level,
            'throttle_events': self.stats['throttle_events'],
            'peak_cpu': self.stats['peak_cpu'],
            'total_throttle_ms': self.stats['total_throttle_ms'],
            'last_throttle': self.stats['last_throttle_time'],
        }


# Global instance
_limiter: Optional[CPULimiter] = None


def get_cpu_limiter() -> CPULimiter:
    """Get or create the global CPU limiter."""
    global _limiter
    if _limiter is None:
        _limiter = CPULimiter()
    return _limiter
