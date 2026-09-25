"""OMLX market direction analyzer for entry/exit decisions.

Queries OMLX to analyze market conditions and decide whether to go long/short/wait
before placing each deal. Can override the bot's configured side based on current
market conditions.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from symbot_python.exchange.base import ExchangeClient, Ticker
from symbot_python.exchange.omlx_advisor import get_advisor

logger = logging.getLogger(__name__)

Direction = Literal["long", "short", "wait"]


async def get_market_data_for_analysis(
    exchange: ExchangeClient,
    symbol: str,
) -> dict:
    """Gather market data needed for OMLX direction analysis."""
    try:
        ticker = await exchange.get_ticker(symbol)
        precision = await exchange.get_precision(symbol)

        # Get klines for recent price action (14 hourly candles = ~14 hours)
        klines = await exchange.get_kline(symbol, "1", limit=14)

        data = {
            "symbol": symbol,
            "current_price": ticker.last,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "spread_bps": (ticker.ask - ticker.bid) / ticker.last * 10000 if ticker.last else 0,
            "volume_24h": ticker.volume_24h_base,
            "turnover_24h": ticker.turnover_24h_quote,
        }

        # Calculate simple trend from klines
        if len(klines) >= 2:
            closes = [k[4] for k in klines]  # close price
            recent_closes = closes[-5:]  # last 5 candles
            price_trend = "up" if recent_closes[-1] > recent_closes[0] else "down"
            volatility = max(closes) - min(closes)
            volatility_pct = (volatility / ticker.last * 100) if ticker.last else 0

            data.update({
                "trend": price_trend,
                "volatility_pct": volatility_pct,
                "highest_24h": max(closes),
                "lowest_24h": min(closes),
            })

        # Get balance info
        balance = await exchange.get_balance()
        data["available_balance"] = balance.get("USDT", 0)

        return data
    except Exception as exc:
        logger.warning("Failed to gather market data: %s", exc)
        return {}


async def analyze_direction(
    exchange: ExchangeClient,
    symbol: str,
    configured_side: str,
) -> Direction:
    """Query OMLX to analyze market and decide if safe to trade, and in which direction.

    Returns:
    - "long": safe to place long order
    - "short": safe to place short order
    - "wait": conditions are risky, skip this cycle
    """
    try:
        market_data = await get_market_data_for_analysis(exchange, symbol)
        if not market_data:
            logger.warning("Could not gather market data, waiting")
            return "wait"

        advisor = await get_advisor()

        # Build context for OMLX analysis
        context = (
            f"Market Analysis for {market_data.get('symbol', symbol)}:\n"
            f"Current Price: ${market_data.get('current_price', '?'):.2f}\n"
            f"Bid-Ask Spread: {market_data.get('spread_bps', 0):.1f} bps\n"
            f"Recent Trend: {market_data.get('trend', 'unknown')}\n"
            f"24h Volatility: {market_data.get('volatility_pct', 0):.2f}%\n"
            f"24h Volume: {market_data.get('volume_24h', 0):.2f}\n"
            f"Available Balance: ${market_data.get('available_balance', 0):.2f}\n"
            f"Configured Side: {configured_side}\n\n"
            f"Based on this analysis, should I trade LONG, SHORT, or WAIT? "
            f"Return only one word: 'long', 'short', or 'wait'."
        )

        response = await advisor._query(context)

        if not response:
            logger.warning("OMLX unavailable, using configured side: %s", configured_side)
            return configured_side  # type: ignore

        # Parse response - look for the direction keyword
        response_lower = response.lower().strip()

        if "short" in response_lower:
            direction = "short"
        elif "long" in response_lower:
            direction = "long"
        elif "wait" in response_lower:
            direction = "wait"
        else:
            logger.warning("OMLX response unclear: %s, using configured side", response[:100])
            return configured_side  # type: ignore

        if direction != configured_side and direction != "wait":
            logger.info(
                "OMLX overriding configured side %s with %s based on market analysis",
                configured_side, direction,
            )

        return direction

    except Exception as exc:
        logger.warning("OMLX direction analysis failed: %s, using configured side", exc)
        return configured_side  # type: ignore
