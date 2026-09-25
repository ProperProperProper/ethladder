"""Shared Bybit public websocket ticker feed — replaces per-tick REST
polling (get_tickers) with ONE persistent connection per symbol, pushed
roughly every 100ms, instead of every open deal's tick loop making its
own fresh HTTPS round-trip every TICK_INTERVAL_SEC. That REST-polling
pattern was directly implicated in a real recurring bug: Bybit (or a
proxy in front of it) periodically resets one of those short-lived
connections (ConnectionResetError), which used to get treated as a
fatal engine crash and force-close a perfectly healthy deal. A single
long-lived websocket connection removes the repeated connection churn
that triggers this, and is lower latency besides.

PaperExchangeClient.get_ticker() reads the cached price from here
instead of making an HTTP call at all once the feed is warm — REST
stays as the fallback for the brief window before the first message
arrives, or if the feed ever goes stale, so nothing regresses if the
websocket is ever slow to (re)connect.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from pybit.unified_trading import WebSocket

from symbot_python.exchange.base import Ticker

logger = logging.getLogger(__name__)

CATEGORY = "linear"
# A message arrives roughly every 100ms while connected; anything older
# than this means the feed is down/reconnecting, so callers should fall
# back to REST rather than trust a frozen price indefinitely.
STALE_AFTER_SEC = 15.0


class TickerStream:
    """One instance per exchange client. Thread-safe: pybit delivers
    messages on its own background thread (separate from asyncio's event
    loop thread), while get_price()/ensure_subscribed() are called from
    the event loop thread.
    """

    def __init__(self) -> None:
        self._ws: Optional[WebSocket] = None
        self._lock = threading.Lock()
        self._subscribed: set[str] = set()
        self._last: dict[str, tuple[Ticker, float]] = {}

    def ensure_subscribed(self, symbol: str) -> None:
        """Idempotent — cheap to call on every single get_ticker(), so
        callers don't need to track subscription state themselves.
        """
        with self._lock:
            if symbol in self._subscribed:
                return
            if self._ws is None:
                # retries=0 means pybit reconnects indefinitely rather
                # than giving up after N attempts — this has to survive
                # for the life of the process, same as the process itself.
                # testnet=False has no default in pybit's own signature
                # (a required kwarg on the parent _WebSocketManager) —
                # omitting it is a TypeError at construction time, only
                # ever surfaced by actually constructing a real one; the
                # mocked unit tests can't catch this by design.
                self._ws = WebSocket(channel_type=CATEGORY, testnet=False, retries=0)
            self._ws.ticker_stream(symbol=symbol, callback=self._on_message)
            self._subscribed.add(symbol)
            logger.info("ticker_stream: subscribed to %s", symbol)

    def _on_message(self, message: dict) -> None:
        # pybit already merges each delta into a running full snapshot
        # per topic before invoking this callback (see
        # _V5WebSocketManager._process_delta_ticker), so `data` here is
        # always the complete current ticker state, not a partial delta.
        try:
            data = message.get("data") or {}
            symbol = data.get("symbol")
            last_price = data.get("lastPrice")
            if not symbol or not last_price:
                return
            ticker = Ticker(
                symbol=symbol,
                last=float(last_price),
                bid=float(data.get("bid1Price") or 0),
                ask=float(data.get("ask1Price") or 0),
                volume_24h_base=float(data.get("volume24h") or 0),
                turnover_24h_quote=float(data.get("turnover24h") or 0),
            )
        except Exception:
            logger.exception("ticker_stream: failed to parse a ticker message")
            return
        with self._lock:
            self._last[symbol] = (ticker, time.monotonic())

    def get_price(self, symbol: str) -> Optional[Ticker]:
        with self._lock:
            entry = self._last.get(symbol)
        if entry is None:
            return None
        ticker, seen_at = entry
        if time.monotonic() - seen_at > STALE_AFTER_SEC:
            return None
        return ticker

    def stop(self) -> None:
        """Synchronous and briefly blocking (pybit's own exit() busy-waits
        for the socket to actually close) — callers on an asyncio event
        loop should run this via asyncio.to_thread rather than await it
        directly.
        """
        with self._lock:
            ws, self._ws = self._ws, None
            self._subscribed.clear()
            self._last.clear()
        if ws is not None:
            try:
                ws.exit()
            except Exception:
                logger.exception("ticker_stream: error while closing websocket")
