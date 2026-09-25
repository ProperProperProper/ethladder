import logging
from pathlib import Path

import pytest

from symbot_python.logging_setup import MelbourneFormatter, configure_logging, log_file_unless_testing


@pytest.fixture(autouse=True)
def reset_root_logger():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in root.handlers:
        h.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def test_log_file_unless_testing_returns_none_under_pytest():
    # We ARE running under pytest right now, so this must always be None
    # here — real entry points (app.py etc.) rely on exactly this to
    # avoid writing test-triggered log lines into the real project logs.
    assert log_file_unless_testing("/some/real/logs/uvicorn.log") is None


def test_configure_logging_is_idempotent():
    configure_logging()
    handlers_after_first = list(logging.getLogger().handlers)
    configure_logging()
    assert logging.getLogger().handlers == handlers_after_first


def test_melbourne_formatter_includes_date_time_and_zone_abbreviation():
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    record.created = 1_700_000_000  # a fixed, real point in time
    formatted = MelbourneFormatter().formatTime(record)
    # e.g. "2023-11-15 06:13:20 AEDT" — must have a real date, a real
    # time, and a zone abbreviation, not just a bare number.
    assert len(formatted.split(" ")) == 3
    date_part, time_part, zone_part = formatted.split(" ")
    assert date_part.count("-") == 2
    assert time_part.count(":") == 2
    assert zone_part in ("AEST", "AEDT")


def test_configure_logging_writes_to_the_given_log_file(tmp_path):
    log_path = tmp_path / "app.log"
    configure_logging(log_file=log_path)
    logging.getLogger("test.module").warning("something happened")
    for h in logging.getLogger().handlers:
        h.flush()

    content = log_path.read_text()
    assert "something happened" in content
    assert "WARNING" in content
    assert ("AEST" in content) or ("AEDT" in content)


def test_configure_logging_second_call_with_a_new_log_file_still_adds_it(tmp_path):
    # Regression: the exact real bug this guards against — app.py's
    # module-level configure_logging() (no log_file) runs first because
    # continuous_optimizer.py imports fetch_klines from it, BEFORE
    # continuous_optimizer.py makes its own configure_logging(log_file=...)
    # call. A naive "already configured once, skip everything" idempotency
    # guard would silently drop the optimizer's own FileHandler here.
    first_log = tmp_path / "web.log"
    second_log = tmp_path / "optimizer.log"

    configure_logging()  # simulates app.py's no-log_file call
    configure_logging(log_file=second_log)  # simulates continuous_optimizer.py's call

    logging.getLogger("test.module").error("optimizer-side error")
    for h in logging.getLogger().handlers:
        h.flush()

    assert not first_log.exists()  # nothing ever asked for this path
    assert second_log.exists()
    assert "optimizer-side error" in second_log.read_text()


def test_configure_logging_does_not_duplicate_a_handler_for_the_same_file(tmp_path):
    log_path = tmp_path / "app.log"
    configure_logging(log_file=log_path)
    configure_logging(log_file=log_path)

    # Filter to a handler for THIS test's exact path — the root logger
    # may already carry unrelated handlers by the time this test runs
    # (pytest's own log-capture plugin, and/or a real one from app.py's
    # module-level configure_logging() call against the project's actual
    # logs/uvicorn.log, since importing that module anywhere earlier in
    # the whole test session already ran it).
    resolved = log_path.resolve()
    matching = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved
    ]
    assert len(matching) == 1
