import pytest
from fastapi.testclient import TestClient

from symbot_python.api import app as app_module
from symbot_python.api import winners as winners_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect to a throwaway temp DB so this never touches the real
    # param library the continuous optimizer may be writing to.
    from symbot_python.strategy import optimization_store

    monkeypatch.setattr(
        winners_module, "connect",
        lambda: optimization_store.connect(tmp_path / "test_optimization_results.db"),
    )
    return TestClient(app_module.app)


def test_winners_page_loads_with_no_data(client):
    response = client.get("/winners")
    assert response.status_code == 200
    assert "No winners recorded yet" in response.text
    assert "class=\"no-data\" style=\"color:#c62828" not in response.text


def test_winners_page_survives_a_db_error_instead_of_500ing(client, monkeypatch):
    # Regression: winners_page() used to have no try/except at all — a
    # transient sqlite error (e.g. "database is locked" from the
    # concurrently-running continuous optimizer) would both leak the
    # connection (never closed) and crash the whole page with a 500.
    def broken_connect():
        raise RuntimeError("simulated database is locked")

    monkeypatch.setattr(winners_module, "connect", broken_connect)
    response = client.get("/winners")
    assert response.status_code == 200
    assert "Could not load the param library" in response.text
