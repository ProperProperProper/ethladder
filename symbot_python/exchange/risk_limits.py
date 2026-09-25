"""Per-symbol maintenance margin rate lookup for Bybit linear perpetuals,
needed for accurate liquidation-price estimation under leverage.

Mirrors the pattern already proven in the user's live ETH bot
(unified-combo-grid's `get_liquidation_frac`): fetch the LOWEST-risk
tier's maintenanceMargin from Bybit's get_risk_limit, cache it per
symbol for the process lifetime, and fall back to a conservative
constant only if the API call fails. Never hardcode one rate across all
symbols — Bybit's actual lowest-tier MMR varies (e.g. BTC/ETH = 0.0033,
SOL = 0.005 at the time this was checked).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Used ONLY if the API call fails — matches the reference bot's fallback
# constant. Prefer the live-fetched, per-symbol value whenever possible.
FALLBACK_MAINTENANCE_MARGIN_RATE = 0.005

_cache: dict[str, float] = {}


class RiskLimitSession(Protocol):
    def get_risk_limit(self, **kwargs: Any) -> dict: ...


def get_maintenance_margin_rate(
    session: RiskLimitSession, symbol: str, category: str = "linear"
) -> float:
    if symbol in _cache:
        return _cache[symbol]
    try:
        response = session.get_risk_limit(category=category, symbol=symbol)
        tiers = response["result"]["list"]
        lowest = next((t for t in tiers if int(t.get("isLowestRisk", 0)) == 1), tiers[0])
        rate = float(lowest["maintenanceMargin"])
        _cache[symbol] = rate
        return rate
    except Exception as exc:
        logger.warning("get_risk_limit failed for %s, using fallback MMR: %s", symbol, exc)
        return FALLBACK_MAINTENANCE_MARGIN_RATE
