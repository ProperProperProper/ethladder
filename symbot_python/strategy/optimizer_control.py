"""Small local control channel for the continuous optimizer.

The web bot and optimizer are separate launchd jobs.  A file request keeps
them decoupled while allowing a web-bot restart to wake an optimizer that is
otherwise in its normal between-cycle sleep.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


REQUEST_PATH = Path(__file__).resolve().parents[2] / "data" / "optimizer_run.request"


def request_optimizer_run() -> None:
    """Atomically request one fresh optimizer cycle, coalescing repeats."""
    REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REQUEST_PATH.with_suffix(".tmp")
    temporary.write_text(f"{time.time():.6f}\n", encoding="utf-8")
    os.replace(temporary, REQUEST_PATH)


def consume_optimizer_run_request() -> bool:
    """Consume a pending request. Missing requests are the normal case."""
    try:
        REQUEST_PATH.unlink()
    except FileNotFoundError:
        return False
    return True
