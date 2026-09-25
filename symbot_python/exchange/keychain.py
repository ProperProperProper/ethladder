"""Read-only real-balance lookup from macOS Keychain credentials.

STRICTLY READ-ONLY: this module calls get_wallet_balance and nothing
else. It must never grow a code path that places, amends, or cancels an
order — that would silently turn "check my real balance" into a live
trading capability, which is explicitly not authorized here (see
exchange/factory.py's confirm_live gate, which this module never calls).

Credentials are read from Keychain via the `security` CLI, used
in-memory to construct a pybit session, and discarded immediately. They
are never printed, logged, or returned to any caller — only the
resulting balance figures leave this module.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass

from pybit.unified_trading import HTTP

# Neither the Keychain subprocess nor the wallet-balance HTTP call had a
# timeout before this — a stalled call here would hang whatever awaited
# it forever (paper trading's get_manager() on first use / every 2h
# refresh, or the continuous optimizer's every-cycle balance fetch),
# same class of bug found and fixed in dca_bot.py's tick loop this session.
KEYCHAIN_READ_TIMEOUT_SEC = 10.0
BALANCE_CALL_TIMEOUT_SEC = 15.0


@dataclass
class RealBalance:
    total_equity: float
    total_available_balance: float
    coin_balances: dict[str, float]


def _read_keychain_json(service: str, account: str) -> dict:
    raw = subprocess.check_output(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        stderr=subprocess.DEVNULL,
        timeout=KEYCHAIN_READ_TIMEOUT_SEC,
    ).decode().strip()
    return json.loads(raw)


async def fetch_real_balance(
    service: str = "unified-combo-grid", account: str = "live", account_type: str = "UNIFIED",
) -> RealBalance:
    """Read-only wallet balance check. Constructs an authenticated pybit
    session from Keychain credentials, calls get_wallet_balance exactly
    once, and discards the credentials — never placing, amending, or
    cancelling anything.
    """
    creds = await asyncio.wait_for(
        asyncio.to_thread(_read_keychain_json, service, account), timeout=KEYCHAIN_READ_TIMEOUT_SEC + 2.0
    )
    session = HTTP(api_key=creds["api_key"], api_secret=creds["api_secret"])
    del creds  # never held longer than needed to construct the client

    response = await asyncio.wait_for(
        asyncio.to_thread(session.get_wallet_balance, accountType=account_type),
        timeout=BALANCE_CALL_TIMEOUT_SEC,
    )
    del session  # this session is single-use; never reused for anything else

    ret_code = response.get("retCode", 0)
    if ret_code != 0:
        raise RuntimeError(f"get_wallet_balance failed: {response.get('retMsg')}")

    accounts = response["result"]["list"]
    if not accounts:
        return RealBalance(total_equity=0.0, total_available_balance=0.0, coin_balances={})

    entry = accounts[0]
    coin_balances = {
        c["coin"]: float(c.get("walletBalance") or 0)
        for c in entry.get("coin", [])
    }
    return RealBalance(
        total_equity=float(entry.get("totalEquity") or 0),
        total_available_balance=float(entry.get("totalAvailableBalance") or 0),
        coin_balances=coin_balances,
    )
