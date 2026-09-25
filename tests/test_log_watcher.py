import time

import run_everything as log_watcher
from run_everything import (
    WATCHER_MAX_TRACEBACK_LINES as MAX_TRACEBACK_LINES,
    Deduper,
    classify_line,
    is_traceback_continuation,
    persist_alert,
    split_into_alert_blocks,
)


def test_classify_line_detects_our_own_log_format():
    assert classify_line("2026-09-16 18:46:46 AEST ERROR continuous_optimizer: boom") == "ERROR"
    assert classify_line("2026-09-16 18:46:46 AEST WARNING continuous_optimizer: careful") == "WARNING"
    assert classify_line("2026-09-16 18:46:46 AEST INFO continuous_optimizer: all fine") is None


def test_classify_line_detects_uvicorn_native_format():
    assert classify_line("ERROR:    something broke") == "ERROR"
    assert classify_line("WARNING:  heads up") == "WARNING"
    assert classify_line('INFO:     127.0.0.1 - "GET /paper HTTP/1.1" 200 OK') is None


def test_classify_line_detects_a_bare_traceback_start():
    assert classify_line("Traceback (most recent call last):") == "ERROR"


def test_classify_line_ignores_blank_lines():
    assert classify_line("") is None
    assert classify_line("   ") is None


def test_is_traceback_continuation():
    assert is_traceback_continuation('  File "dca_bot.py", line 10, in _tick') is True
    assert is_traceback_continuation("RuntimeError: boom") is True
    assert is_traceback_continuation("") is True
    assert is_traceback_continuation("2026-09-16 18:46:46 AEST INFO x: next entry") is False
    assert is_traceback_continuation("ERROR:    a new uvicorn-native line") is False


def test_deduper_suppresses_repeats_within_the_window():
    dedup = Deduper(window_sec=100.0)
    assert dedup.should_alert("same-key") is True
    assert dedup.should_alert("same-key") is False  # too soon
    assert dedup.should_alert("different-key") is True  # unrelated key, unaffected


def test_deduper_allows_repeat_after_window_elapses():
    dedup = Deduper(window_sec=0.05)
    assert dedup.should_alert("k") is True
    time.sleep(0.1)
    assert dedup.should_alert("k") is True


def test_persist_alert_writes_to_sqlite(tmp_path, monkeypatch):
    from symbot_python.strategy import watcher_store

    db_path = tmp_path / "test_watcher_alerts.db"
    monkeypatch.setattr(log_watcher, "connect_watcher_db", lambda: watcher_store.connect(db_path))

    persist_alert("optimizer", "ERROR", "something broke")

    conn = watcher_store.connect(db_path)
    try:
        rows = watcher_store.recent_alerts(conn)
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["message"] == "something broke"


def test_persist_alert_never_raises_on_a_db_failure(monkeypatch):
    def broken_connect():
        raise RuntimeError("simulated db failure")

    monkeypatch.setattr(log_watcher, "connect_watcher_db", broken_connect)
    persist_alert("web", "ERROR", "should not raise")  # must not raise


def test_persist_alert_survives_a_real_sqlite_lock(tmp_path, monkeypatch):
    """The docstring claims a DB lock/contention failure is caught and
    non-fatal — prove it against a REAL sqlite3 lock (a second connection
    holding BEGIN EXCLUSIVE), not just a generically mocked exception, so
    the actual sqlite3.OperationalError code path is exercised. The
    default 5s busy-timeout is patched down so the test doesn't take 5s
    to prove the point.
    """
    import sqlite3

    from symbot_python.strategy import watcher_store

    db_path = tmp_path / "locked.db"
    watcher_store.connect(db_path).close()  # create the schema first

    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN EXCLUSIVE")

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: real_connect(*a, **{**kw, "timeout": 0.05})
    )
    monkeypatch.setattr(log_watcher, "connect_watcher_db", lambda: watcher_store.connect(db_path))

    try:
        persist_alert("optimizer", "ERROR", "should survive a real lock")  # must not raise
    finally:
        blocker.rollback()
        blocker.close()


# --- split_into_alert_blocks: traceback-collapsing edge cases ---------


def test_traceback_at_end_of_file_with_no_trailing_content_is_fully_collected():
    lines = [
        "2026-09-16 18:00:00 AEST ERROR mod: boom",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: boom",
    ]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 1
    severity, first_line, text = blocks[0]
    assert severity == "ERROR"
    assert text == "\n".join(lines)


def test_two_tracebacks_separated_by_one_blank_line_are_not_merged():
    # Regression: a bare "Traceback (most recent call last):" line has no
    # timestamp, so is_traceback_continuation() alone can't distinguish a
    # SECOND, separate traceback from a continuation of the first one —
    # before the fix, these merged into a single over-long alert block.
    lines = [
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: first",
        "",
        "Traceback (most recent call last):",
        '  File "y.py", line 2, in <module>',
        "TypeError: second",
    ]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 2
    assert "ValueError: first" in blocks[0][2]
    assert "TypeError: second" not in blocks[0][2]
    assert "TypeError: second" in blocks[1][2]
    assert "ValueError: first" not in blocks[1][2]


def test_two_back_to_back_tracebacks_with_no_blank_line_are_not_merged():
    # Same bug, more common shape: two exceptions logged one right after
    # the other with no blank line at all between them.
    lines = [
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: first",
        "Traceback (most recent call last):",
        '  File "y.py", line 2, in <module>',
        "TypeError: second",
    ]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 2
    assert "ValueError: first" in blocks[0][2]
    assert "TypeError: second" in blocks[1][2]


def test_a_tracebacks_own_header_line_stays_with_its_trigger_line():
    # The normal, most common case: a log record's own message line
    # immediately followed by ITS OWN "Traceback (most recent call
    # last):" header must stay in the SAME block, not get split off —
    # only a traceback header appearing deeper within an already-started
    # block signals a genuinely new, separate traceback.
    lines = [
        "2026-09-16 18:00:00 AEST ERROR mod: something failed",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: boom",
    ]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0][2] == "\n".join(lines)


def test_exception_line_is_not_dropped_before_a_following_timestamped_entry():
    lines = [
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: boom",
        "2026-09-16 18:00:01 AEST INFO mod: next entry",
    ]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0][2] == "\n".join(lines[:3])


def test_traceback_exceeding_the_cap_does_not_produce_a_bogus_second_alert():
    # A traceback deeper than MAX_TRACEBACK_LINES gets truncated in the
    # alert text, but the leftover raw continuation lines must not be
    # misread as a second, confusing alert — classify_line() only fires
    # on an ALL-CAPS whole-word CRITICAL/ERROR/WARNING or a fresh
    # "Traceback (most recent call last):" line, neither of which a
    # plain indented "File ..." frame line matches.
    frame_lines = [f'  File "x.py", line {n}, in frame_{n}' for n in range(60)]
    lines = ["Traceback (most recent call last):"] + frame_lines + ["ValueError: boom"]
    blocks = split_into_alert_blocks(lines)
    assert len(blocks) == 1
    severity, first_line, text = blocks[0]
    assert len(text.splitlines()) == MAX_TRACEBACK_LINES
    # the final exception line, past the cap, is simply not part of this
    # alert's text — not corrupted, not duplicated as a second alert.
    assert "ValueError: boom" not in text
