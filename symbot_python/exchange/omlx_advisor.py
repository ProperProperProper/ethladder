"""Advisory layer for Bybit trading best practices via local OMLX.

OMLX provides reference guidance on fees, risk management, and order execution
without changing the bot's autonomous direct-to-Bybit architecture. Query it
for learning, validation, and optimization hints — the bot executes its own
decisions via BybitClient.
"""

from __future__ import annotations

import asyncio
import httpx
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

OMLX_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
OMLX_MODEL = "claude-3-5-sonnet-20241022"
# Low temperature for consistent, deterministic advice
OMLX_TEMPERATURE = 0.2
# Fast timeout since this is advisory, not critical path
OMLX_TIMEOUT = 30.0  # Claude responses can take a while locally


class OMLXAdvisor:
    """Query local OMLX for Bybit trading best practices.

    Never blocks order placement — advice is optional and advisory only.
    If OMLX is unavailable, bot continues without it (graceful degradation).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._client: Optional[httpx.AsyncClient] = None

    async def _query(self, question: str) -> str | None:
        """Ask OMLX a question, return the response or None if unavailable."""
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=OMLX_TIMEOUT) as client:
                response = await client.post(
                    OMLX_ENDPOINT,
                    json={
                        "model": OMLX_MODEL,
                        "messages": [{"role": "user", "content": question}],
                        "temperature": OMLX_TEMPERATURE,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("OMLX advisor unavailable: %s (continuing without advice)", exc)
        return None

    async def ask_fee_optimization(self, symbol: str, qty: float, side: str) -> Optional[str]:
        """Ask OMLX how to minimize fees for this order."""
        question = (
            f"Best way to minimize fees on Bybit for a {side} {qty} {symbol} order? "
            "Should I use limit or market? Any rebate opportunities?"
        )
        return await self._query(question)

    async def ask_order_type(self, symbol: str, qty: float, volatility: str = "normal") -> Optional[str]:
        """Ask OMLX whether to use limit or market for this order."""
        question = (
            f"For a {qty} {symbol} order in {volatility} market conditions, "
            "is limit or market order better? Why? Any slippage concerns?"
        )
        return await self._query(question)

    async def ask_position_sizing(self, leverage: float, equity: float, market_condition: str = "normal") -> Optional[str]:
        """Ask OMLX about safe position sizing given risk parameters."""
        question = (
            f"Given {equity:.2f} USDT equity and {leverage}x leverage on ETHUSDT "
            f"in {market_condition} market, what's a safe position size? "
            "Safety margin for liquidation?"
        )
        return await self._query(question)

    async def ask_stop_loss(self, entry_price: float, leverage: float, mmr: float) -> Optional[str]:
        """Ask OMLX for optimal stop-loss placement."""
        question = (
            f"For a long position at {entry_price} with {leverage}x leverage "
            f"and {mmr:.2%} maintenance margin rate, where should I place stop-loss? "
            "What's the liquidation price?"
        )
        return await self._query(question)

    async def ask_funding_info(self, symbol: str) -> Optional[str]:
        """Ask OMLX about current funding rates and settlement times."""
        question = (
            f"What's the current {symbol} perpetual funding rate on Bybit? "
            "When does it settle? Any patterns in funding?"
        )
        return await self._query(question)

    async def validate_risk(
        self,
        symbol: str,
        leverage: float,
        entry_price: float,
        position_qty: float,
        mmr: float,
    ) -> Optional[str]:
        """Validate that a trade meets risk criteria."""
        notional = entry_price * position_qty
        question = (
            f"Is this trade safe? {symbol} at {entry_price}, "
            f"{position_qty} qty ({notional:.2f} notional), "
            f"{leverage}x leverage, {mmr:.2%} MMR. Liquidation price and buffer?"
        )
        return await self._query(question)

    async def ask_api_gotcha(self, context: str) -> Optional[str]:
        """Ask OMLX about common Bybit API pitfalls or edge cases."""
        question = (
            f"Any common gotchas or edge cases with Bybit API for this scenario: {context}? "
            "What should I watch out for?"
        )
        return await self._query(question)

    async def close(self) -> None:
        """Clean up resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Global advisor instance
_advisor: Optional[OMLXAdvisor] = None


async def get_advisor() -> OMLXAdvisor:
    """Get or create the global OMLX advisor."""
    global _advisor
    if _advisor is None:
        _advisor = OMLXAdvisor(enabled=True)
    return _advisor
