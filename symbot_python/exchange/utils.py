"""Shared helpers for talking to Bybit's v5 API shape, used by both the
live and paper clients.
"""

from __future__ import annotations

import asyncio
import math

# No exchange call anywhere in this codebase had a timeout before this —
# one stalled network request (a real, observed failure mode: pybit's
# underlying synchronous HTTP call can block indefinitely on a network
# stall) could hang whatever awaited it forever, with no exception and
# no log line to explain why. Every asyncio.to_thread(...) call in
# bybit_client.py/paper_client.py is wrapped through call_with_timeout so
# a stall surfaces as asyncio.TimeoutError instead of hanging the caller
# (a deal's tick loop, a page render, an optimizer cycle) indefinitely.
EXCHANGE_HTTP_TIMEOUT_SEC = 10.0


async def call_with_timeout(awaitable, timeout: float = EXCHANGE_HTTP_TIMEOUT_SEC):
    """Bounds one exchange call. Note: asyncio.wait_for only gives up on
    AWAITING the call — it cannot forcibly kill the OS thread a
    to_thread(...)-wrapped synchronous HTTP call actually runs in (a
    fundamental Python limitation, not something fixable here). Still
    strictly better than no bound: the caller gets control back and can
    retry or fail cleanly instead of hanging forever.
    """
    return await asyncio.wait_for(awaitable, timeout=timeout)


class BybitApiError(RuntimeError):
    def __init__(self, ret_code: int, ret_msg: str):
        super().__init__(f"Bybit API error {ret_code}: {ret_msg}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg


def unwrap(response: dict) -> dict:
    ret_code = response.get("retCode", 0)
    if ret_code != 0:
        raise BybitApiError(ret_code, response.get("retMsg", "unknown error"))
    return response.get("result", {})


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    decimals = max(0, decimals_for_step(step))
    steps = math.floor(value / step + 1e-9)
    return round(steps * step, decimals)


def decimals_for_step(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    if "." in text:
        return len(text.split(".", 1)[1])
    return 0


def format_qty(qty: float) -> str:
    # Bybit expects qty as a plain decimal string with no scientific
    # notation and no trailing zeros beyond what the instrument allows.
    text = f"{qty:.10f}".rstrip("0").rstrip(".")
    return text or "0"
