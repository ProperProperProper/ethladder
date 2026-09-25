from symbot_python.strategy.signal_bot import is_api_start


def test_is_api_start_true():
    assert is_api_start(["api"]) is True
    assert is_api_start(["API"]) is True


def test_is_api_start_false_for_asap_and_signal():
    assert is_api_start(["asap"]) is False
    assert is_api_start(["signal|3cqs|xyz"]) is False
    assert is_api_start([]) is False
