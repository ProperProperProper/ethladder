import json

import pytest

from symbot_python.exchange import keychain


class FakeSession:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def get_wallet_balance(self, **kwargs):
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "totalEquity": "5000.1234",
                        "totalAvailableBalance": "5000.1234",
                        "coin": [
                            {"coin": "USDT", "walletBalance": "5000.5678"},
                            {"coin": "BTC", "walletBalance": "0"},
                        ],
                    }
                ]
            },
        }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        keychain.subprocess, "check_output",
        lambda *a, **k: json.dumps({"api_key": "fake_key", "api_secret": "fake_secret"}).encode(),
    )
    monkeypatch.setattr(keychain, "HTTP", FakeSession)


async def test_fetch_real_balance_parses_response(patched):
    balance = await keychain.fetch_real_balance()
    assert balance.total_equity == pytest.approx(5000.1234)
    assert balance.total_available_balance == pytest.approx(5000.1234)
    assert balance.coin_balances["USDT"] == pytest.approx(5000.5678)


async def test_fetch_real_balance_raises_on_api_error(monkeypatch, patched):
    class ErrorSession(FakeSession):
        def get_wallet_balance(self, **kwargs):
            return {"retCode": 10001, "retMsg": "invalid api key", "result": {}}

    monkeypatch.setattr(keychain, "HTTP", ErrorSession)
    with pytest.raises(RuntimeError, match="invalid api key"):
        await keychain.fetch_real_balance()


async def test_fetch_real_balance_empty_accounts_returns_zero(monkeypatch, patched):
    class EmptySession(FakeSession):
        def get_wallet_balance(self, **kwargs):
            return {"retCode": 0, "retMsg": "OK", "result": {"list": []}}

    monkeypatch.setattr(keychain, "HTTP", EmptySession)
    balance = await keychain.fetch_real_balance()
    assert balance.total_equity == 0.0
    assert balance.coin_balances == {}


async def test_fetch_real_balance_times_out_instead_of_hanging_forever(monkeypatch, patched):
    import asyncio
    import time

    monkeypatch.setattr(keychain, "BALANCE_CALL_TIMEOUT_SEC", 0.05)

    class HangingSession(FakeSession):
        def get_wallet_balance(self, **kwargs):
            # Long enough to guarantee it outlasts the 0.05s timeout above
            # (proving wait_for actually gives up rather than blocking),
            # short enough that the real OS thread this runs in (asyncio.
            # to_thread can't be cancelled) doesn't hold up process exit.
            time.sleep(1.0)
            return super().get_wallet_balance(**kwargs)

    monkeypatch.setattr(keychain, "HTTP", HangingSession)
    with pytest.raises(asyncio.TimeoutError):
        await keychain.fetch_real_balance()


def test_credentials_never_appear_in_the_returned_object(patched):
    import asyncio

    balance = asyncio.run(keychain.fetch_real_balance())
    dumped = str(vars(balance))
    assert "fake_key" not in dumped
    assert "fake_secret" not in dumped
