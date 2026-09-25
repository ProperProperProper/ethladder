"""Independent paper-only account mirroring confirmed original paper fills."""

from __future__ import annotations

import time
from dataclasses import dataclass

from symbot_python.exchange.paper_client import PaperExchangeClient, PaperFill


@dataclass(frozen=True)
class MirrorFill:
    timestamp: float
    source_order_id: str
    mirror_order_id: str | None
    symbol: str
    side: str
    qty: float
    source_price: float
    mirror_price: float
    error: str | None = None


class ReversePaper:
    """Never submits live orders; a separate PaperExchangeClient owns its wallet."""

    def __init__(self, source: PaperExchangeClient, initial_balance: float):
        self.client = PaperExchangeClient(
            source._session, initial_balances={"USDT": initial_balance},
            fee_rate_percent=source.fee_rate_percent,
        )
        self.initial_balance = initial_balance
        self.fills: list[MirrorFill] = []
        source.on_fill = self.mirror
        source.on_funding = self.mirror_funding

    def mirror(self, fill: PaperFill) -> None:
        side = "Sell" if fill.side == "Buy" else "Buy"
        mirror_order_id = None
        error = None
        try:
            result = self.client.place_market_order_at_price(
                fill.symbol, side, fill.qty, fill.opposite_price,
                leverage=fill.leverage,
            )
            mirror_order_id = result.order_id
        except Exception as exc:
            # Source fill already succeeded. Record divergence; never undo it.
            error = f"{type(exc).__name__}: {exc}"
        self.fills.append(MirrorFill(
            time.time(), fill.order_id, mirror_order_id, fill.symbol,
            side, fill.qty, fill.price, fill.opposite_price, error,
        ))

    def mirror_funding(self, source_cost: float) -> None:
        # Opposite signed position has the opposite funding cash flow.
        self.client.apply_funding_cost(-source_cost)
