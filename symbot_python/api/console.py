"""Live log console in the browser — tails logs/run_everything.log (the
ONE process everything runs in as of the 2026-09-23 single-script
merge) and logs/optimizer_activity.log (the optimizer's own dedicated
subset of it — see run_everything.py's optimizer_log) so you can watch
everything (optimizer cycle progress, promotions/demotions, engine
crashes with their Melbourne timestamp — see logging_setup.py) directly
in the web app instead of needing a terminal.

Previously pointed at logs/continuous_optimizer.log and logs/uvicorn.log
— two separate per-process log files from before the merge, both
permanently empty (0 bytes) ever since, since nothing writes to them
any more. The "Continuous Optimizer" tab genuinely showed nothing,
forever, regardless of whether the optimizer itself was working.

A later attempt fixed this by filtering an 8 MiB tail of the combined
run_everything.log for optimizer-tagged lines — still not enough:
OMLX dip-analysis logging during forward-test replay alone was observed
running north of 50 MB/minute, so an 8 MiB window could cover under 10
seconds of real time, nowhere near enough to reliably contain the
optimizer's own genuinely sparse lines (one every few minutes) even
while it was working perfectly. run_everything.py now writes the
optimizer's own lines to a small dedicated file in addition to the
combined log (propagate=True keeps them in run_everything.log too, this
is purely additive) — reading that directly sidesteps the volume
mismatch entirely instead of trying to out-search it.

Read-only: this route never starts/stops/restarts anything, it only
reads whatever's already been written to the log file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from symbot_python.strategy.watcher_store import connect as connect_watcher_db, recent_alerts

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_SOURCES = {
    "optimizer": LOG_DIR / "optimizer_activity.log",
    "web": LOG_DIR / "run_everything.log",
}
# Read at most this many trailing bytes regardless of how large the log
# file has grown over weeks of unattended uptime — tailing must stay
# cheap on every poll, not re-read a multi-hundred-MB file from the start.
MAX_TAIL_BYTES = 262_144  # 256 KiB
DEFAULT_LINES = 300

router = APIRouter()


def tail_lines(path: Path, n_lines: int, max_bytes: int = MAX_TAIL_BYTES) -> list[str]:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard a possibly-partial first line from the seek
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n_lines:]
    except Exception as exc:
        return [f"[console: failed to read {path.name}: {exc}]"]


@router.get("/console", response_class=HTMLResponse)
async def console_page(request: Request):
    return templates.TemplateResponse(request, "console.html", {})


@router.get("/api/logs")
async def api_logs(source: str = "optimizer", lines: int = DEFAULT_LINES) -> dict:
    path = LOG_SOURCES.get(source)
    if path is None:
        return {"error": f"unknown source {source!r}, expected one of {list(LOG_SOURCES)}", "lines": []}
    lines = max(1, min(lines, 2000))
    return {"source": source, "path": str(path), "lines": tail_lines(path, lines)}


@router.get("/api/alerts")
async def api_alerts(limit: int = 50) -> dict:
    """Recent alerts from scripts/log_watcher.py's SQLite record — the
    durable, queryable side of what also fires as a native macOS
    notification. This route never runs the watcher itself, only reads
    whatever it has already recorded.
    """
    limit = max(1, min(limit, 500))
    conn = None
    try:
        conn = connect_watcher_db()
        rows = [dict(r) for r in recent_alerts(conn, limit=limit)]
        return {"alerts": rows}
    except Exception:
        logger.exception("Failed to read watcher alerts.")
        return {"alerts": [], "error": "Could not load alerts right now."}
    finally:
        if conn is not None:
            conn.close()
