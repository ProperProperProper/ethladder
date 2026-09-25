from symbot_python.strategy import optimizer_control


def test_optimizer_run_request_is_consumed_once(tmp_path, monkeypatch):
    monkeypatch.setattr(optimizer_control, "REQUEST_PATH", tmp_path / "optimizer_run.request")

    optimizer_control.request_optimizer_run()

    assert optimizer_control.consume_optimizer_run_request() is True
    assert optimizer_control.consume_optimizer_run_request() is False
