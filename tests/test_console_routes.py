import pytest
from fastapi.testclient import TestClient

from symbot_python.api import app as app_module
from symbot_python.api import console as console_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def fake_log(tmp_path, monkeypatch):
    # "optimizer" reads its own dedicated file (logs/optimizer_activity.log
    # — see run_everything.py's optimizer_log) directly and unfiltered as
    # of this fix; no in-request filtering happens any more (an earlier
    # version filtered a shared combined-log tail, which turned out to be
    # nowhere near big enough — OMLX dip-analysis logging alone was
    # observed running north of 50 MB/minute, so even an 8 MiB tail could
    # cover under 10 seconds of real time).
    log_path = tmp_path / "fake.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")
    monkeypatch.setitem(console_module.LOG_SOURCES, "optimizer", log_path)
    return log_path


def test_console_page_loads(client):
    response = client.get("/console")
    assert response.status_code == 200
    assert "Console" in response.text


def test_api_logs_returns_recent_lines(client, fake_log):
    response = client.get("/api/logs?source=optimizer&lines=3")
    assert response.status_code == 200
    data = response.json()
    assert data["lines"] == ["line 8", "line 9", "line 10"]


def test_api_logs_missing_file_returns_empty(client, tmp_path, monkeypatch):
    monkeypatch.setitem(console_module.LOG_SOURCES, "optimizer", tmp_path / "does_not_exist.log")
    response = client.get("/api/logs?source=optimizer")
    assert response.status_code == 200
    assert response.json()["lines"] == []


def test_api_logs_unknown_source_reports_error_not_500(client):
    response = client.get("/api/logs?source=bogus")
    assert response.status_code == 200
    data = response.json()
    assert data["lines"] == []
    assert "unknown source" in data["error"]


@pytest.fixture
def fake_watcher_db(tmp_path, monkeypatch):
    from symbot_python.strategy import watcher_store

    db_path = tmp_path / "test_watcher_alerts.db"
    monkeypatch.setattr(console_module, "connect_watcher_db", lambda: watcher_store.connect(db_path))
    return db_path


def test_api_alerts_returns_recent_rows(client, fake_watcher_db):
    from symbot_python.strategy import watcher_store

    conn = watcher_store.connect(fake_watcher_db)
    watcher_store.record_alert(conn, "optimizer", "ERROR", "boom")
    conn.close()

    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert len(data["alerts"]) == 1
    assert data["alerts"][0]["message"] == "boom"
    assert data["alerts"][0]["severity"] == "ERROR"


def test_api_alerts_empty_when_none_recorded(client, fake_watcher_db):
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert response.json()["alerts"] == []


def test_api_alerts_survives_a_db_error_instead_of_500ing(client, monkeypatch):
    def broken_connect():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(console_module, "connect_watcher_db", broken_connect)
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert data["alerts"] == []
    assert "error" in data


def test_tail_lines_only_reads_the_tail_of_a_large_file(tmp_path):
    # Regression-style check: tail_lines must not choke on / fully
    # re-read a file much larger than MAX_TAIL_BYTES on every poll.
    big_path = tmp_path / "big.log"
    with big_path.open("w") as f:
        for i in range(50_000):
            f.write(f"line {i}\n")

    result = console_module.tail_lines(big_path, n_lines=5, max_bytes=1024)
    assert result[-1] == "line 49999"
    assert len(result) == 5
