import json

import pytest

import run_everything as co


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    # Never touch the real data/optimizer_status.json — the actual
    # unified process may be running concurrently and writing to it for
    # real.
    path = tmp_path / "optimizer_status.json"
    monkeypatch.setattr(co, "STATUS_PATH", path)
    return path


def test_write_status_creates_valid_json(status_path):
    co.write_optimizer_status(state="fetching", symbol="ETHUSDT")
    data = json.loads(status_path.read_text())
    assert data["state"] == "fetching"
    assert data["symbol"] == "ETHUSDT"
    assert "updated_at" in data


def test_write_status_merges_into_existing_fields(status_path):
    co.write_optimizer_status(state="fetching", symbol="ETHUSDT", cycle=1)
    co.write_optimizer_status(state="searching", combos_done=5)
    data = json.loads(status_path.read_text())
    # symbol/cycle from the first call must survive the second call's update.
    assert data["symbol"] == "ETHUSDT"
    assert data["cycle"] == 1
    assert data["state"] == "searching"
    assert data["combos_done"] == 5


def test_write_status_is_atomic_no_leftover_tmp_file(status_path):
    co.write_optimizer_status(state="sleeping")
    tmp_file = status_path.with_suffix(".json.tmp")
    # The write must go through a temp file + os.replace so a concurrent
    # reader (the web GUI polls this every 3s) can never observe a
    # partially-written file — confirmed by the temp file never
    # surviving a successful write.
    assert not tmp_file.exists()
    assert status_path.exists()
    assert json.loads(status_path.read_text())["state"] == "sleeping"


def test_write_status_never_raises_on_a_corrupted_existing_file(status_path):
    status_path.write_text("{not valid json")
    # Must recover by treating the corrupted file as empty, not crash the
    # caller (this is called from inside the optimizer's main loop).
    co.write_optimizer_status(state="error")
    data = json.loads(status_path.read_text())
    assert data["state"] == "error"


def test_continuous_optimizer_uses_100k_combos_per_walk_forward_window():
    assert co.SEARCH_SAMPLES == 100_000
