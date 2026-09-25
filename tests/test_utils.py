import asyncio

import pytest

from symbot_python.exchange import utils


async def test_call_with_timeout_returns_the_awaitable_result_when_fast():
    async def fast():
        return 42

    assert await utils.call_with_timeout(fast(), timeout=1.0) == 42


async def test_call_with_timeout_raises_when_the_awaitable_stalls():
    # Reproduces the real class of bug this exists to prevent: no
    # exchange call anywhere had a bound before this, so one stalled
    # network request (a real, observed failure mode) could hang
    # whatever awaited it forever — a paper deal's tick loop, a page
    # render, an optimizer cycle — with no exception and no log line.
    async def stalls():
        await asyncio.sleep(3600)

    with pytest.raises(asyncio.TimeoutError):
        await utils.call_with_timeout(stalls(), timeout=0.05)


async def test_call_with_timeout_default_is_a_real_positive_bound():
    assert utils.EXCHANGE_HTTP_TIMEOUT_SEC > 0
