import pytest

from symbot_python.strategy.watcher_store import connect, record_alert, recent_alerts


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test_watcher_alerts.db")
    yield c
    c.close()


def test_record_and_read_back_an_alert(conn):
    alert_id = record_alert(conn, source="optimizer", severity="ERROR", message="boom")
    rows = recent_alerts(conn)
    assert len(rows) == 1
    assert rows[0]["id"] == alert_id
    assert rows[0]["source"] == "optimizer"
    assert rows[0]["severity"] == "ERROR"
    assert rows[0]["message"] == "boom"
    assert "AEST" in rows[0]["created_at"] or "AEDT" in rows[0]["created_at"]


def test_recent_alerts_orders_newest_first(conn):
    record_alert(conn, "web", "WARNING", "first")
    record_alert(conn, "web", "ERROR", "second")
    rows = recent_alerts(conn)
    assert [r["message"] for r in rows] == ["second", "first"]


def test_recent_alerts_respects_limit(conn):
    for i in range(5):
        record_alert(conn, "web", "WARNING", f"msg {i}")
    rows = recent_alerts(conn, limit=2)
    assert len(rows) == 2
    assert rows[0]["message"] == "msg 4"
