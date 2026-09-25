"""Shared logging configuration for the long-running processes (the web
server, the continuous optimizer, the log watcher) — every log line gets
an explicit Australia/Melbourne timestamp with the zone abbreviation
printed (AEST/AEDT), not an ambiguous unlabeled timestamp that depends
on whatever timezone the machine happens to be set to.

Also writes directly to a log file via logging.FileHandler rather than
relying on shell/launchd stdout redirection (`> file` / plist
StandardOutPath) as the only path to disk. This matters specifically for
launchd: testing found that launchd's OWN StandardOutPath/StandardErrorPath
file creation into ~/Documents/... silently produces an empty file for
this project's (Homebrew, ad-hoc-signed) venv Python interpreter, even
though that exact same interpreter's OWN direct open()/write() calls into
the same folder work perfectly fine — apparently a TCC-related distinction
between launchd's redirection machinery and a script's own file I/O, not
a blanket Documents-folder restriction (a trusted binary like /bin/echo
redirected the normal way works fine; a script explicitly opening the
file itself also works fine either way). Writing our own log content via
FileHandler sidesteps the flaky mechanism entirely — a launchd plist's
StandardOutPath/StandardErrorPath is now only a fallback for output from
before configure_logging() runs (e.g. a very early import error).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")


def log_file_unless_testing(path: Path | str) -> Path | None:
    """Real entry points (app.py, continuous_optimizer.py,
    log_watcher.py) call configure_logging(log_file=...) at MODULE
    IMPORT time, and the test suite imports all three — without this,
    every test run would append real (test-triggered) log lines into
    the actual logs/uvicorn.log etc., polluting what the live web
    console page and log watcher show. Callers should wrap their
    production log_file path with this rather than passing it directly.
    Tests that want to exercise configure_logging()'s own file-writing
    behavior call it directly (not through this helper) with a tmp_path,
    so they're unaffected.
    """
    return None if "pytest" in sys.modules else Path(path)


class MelbourneFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=MELBOURNE_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S %Z")


def _make_formatter() -> MelbourneFormatter:
    return MelbourneFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def configure_logging(level: int = logging.INFO, log_file: Path | str | None = None) -> None:
    """Safe to call repeatedly and from multiple entry points that import
    each other (app.py, continuous_optimizer.py, log_watcher.py) — each
    piece (the one stdout handler, each distinct log_file's FileHandler)
    is only ever installed once, so calling this again later with a
    log_file the first call didn't know about still adds that file
    (this matters because continuous_optimizer.py imports fetch_klines
    from app.py, which calls this with no log_file, BEFORE
    continuous_optimizer.py makes its own call with one — a naive
    "already configured, skip everything" guard would silently drop the
    optimizer's own FileHandler).
    """
    root = logging.getLogger()
    root.setLevel(level)

    has_plain_stream_handler = any(
        type(h) is logging.StreamHandler and isinstance(h.formatter, MelbourneFormatter)
        for h in root.handlers
    )
    if not has_plain_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(_make_formatter())
        root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        already_has_this_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path
            for h in root.handlers
        )
        if not already_has_this_file:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(_make_formatter())
            root.addHandler(file_handler)
