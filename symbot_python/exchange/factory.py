"""Exchange client construction. This is the ONE place in the
codebase where a live-capable (real-money) client can be created — never
construct BybitClient directly from bot config or from the web layer.
"""

from __future__ import annotations

import asyncio

from pybit.unified_trading import HTTP

from symbot_python.exchange.base import ExchangeClient, TradingMode
from symbot_python.exchange.bybit_client import BybitClient
from symbot_python.exchange.paper_client import PaperExchangeClient
from symbot_python.exchange.ticker_stream import TickerStream


async def create_exchange_client(
    mode: TradingMode,
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    paper_initial_balances: dict[str, float] | None = None,
    paper_fee_rate_percent: float = 0.1,
) -> ExchangeClient:
    if mode == TradingMode.BACKTEST:
        raise ValueError(
            "Backtest mode doesn't use an ExchangeClient at all — call "
            "strategy.backtest.run_backtest() directly."
        )

    if mode == TradingMode.PAPER:
        public_session = HTTP()  # no api_key/secret: can never place a real order
        return PaperExchangeClient(
            public_session,
            initial_balances=paper_initial_balances,
            fee_rate_percent=paper_fee_rate_percent,
            ticker_stream=TickerStream(),
        )

    if mode == TradingMode.LIVE:
        # If credentials not provided, read from Keychain
        if not api_key or not api_secret:
            try:
                from symbot_python.exchange.keychain import _read_keychain_json
                creds = await asyncio.to_thread(_read_keychain_json, "unified-combo-grid", "live")
                api_key = creds.get("api_key")
                api_secret = creds.get("api_secret")
            except Exception as e:
                raise ValueError(f"Live trading requires credentials in Keychain: {e}")

        if not api_key or not api_secret:
            raise ValueError("Live trading requires api_key and api_secret.")
        session = HTTP(api_key=api_key, api_secret=api_secret)
        client = BybitClient(session)
        await client.verify_connection()  # fail fast on bad credentials, before any bot uses it
        return client

    raise ValueError(f"Unknown trading mode: {mode}")
