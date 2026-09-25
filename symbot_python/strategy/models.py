"""In-memory Bot/Deal representations for the async engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Optional

from symbot_python.strategy.dca_math import OrderRung

Side = Literal["long", "short"]


def to_exchange_symbol(pair: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT' (Bybit spot symbols have no separator)."""
    return pair.replace("/", "").upper()


@dataclass
class BotConfig:
    bot_name: str
    pair: str  # e.g. "BTC/USDT"
    bot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    side: Side = "long"  # "long" = buy dips/sell rallies; "short" = mirror image

    # Stop-and-reverse: when the CURRENT deal's own drawdown (raw
    # price-move %, same convention as dca_stop_loss_percent — negative
    # meaning underwater) reaches this magnitude, close it and
    # immediately open an equal-conviction position in the OPPOSITE
    # direction — one combined market order, not two separate ones (see
    # dca_bot.py's _evaluate_reverse_drawdown/_handle_sell and
    # dca_bot_manager.py's _start_deal_from_flip). None/0 = disabled
    # (default): this is a deliberate strategy choice, not something
    # every bot should silently do. Checked every tick against the
    # position's OWN drawdown — NOT triggered by take_profit/stop_loss/
    # cancel/panic_sell/liquidated closing on their own account.
    reverse_drawdown_percent: Optional[float] = None
    # Hysteresis against whipsaw thrash (price dips, flips, immediately
    # bounces back, the new position is now also underwater, flips
    # again...): no two reversals closer together than this, in seconds.
    reverse_cooldown_sec: float = 3600.0
    # Hard cap on how many reversals can happen back-to-back before the
    # deal just stops reversing and defers to its own normal exits
    # (stop-loss/take-profit/liquidation) instead — bounds the worst
    # case of a genuine sustained whipsaw compounding losses on both
    # sides of the same move. A normal (non-reversal) close resets this
    # streak to 0.
    max_consecutive_reversals: int = 2
    # Runtime counters, not user-facing config — live on BotConfig
    # because it persists across a bot's whole chain of deals (each
    # deal is a fresh object), same pattern as deal_count below.
    consecutive_reversals: int = 0
    last_reversal_ts: float = 0.0

    first_order_amount: float = 20.0
    dca_order_amount: float = 45.0
    dca_max_order: int = 10
    dca_order_size_multiplier: float = 1.08
    dca_order_start_distance: float = 1.3
    dca_order_step_percent: float = 1.3
    dca_order_step_percent_multiplier: float = 1.0
    dca_take_profit_percent: float = 0.33
    exchange_fee: float = 0.1

    pair_deals_max: int = 1  # >1 allows multiple concurrent deals on the same pair for THIS bot
    pair_bots_deals_max: int = 0  # global cap across all bots, 0 = unlimited
    deal_max: int = 0  # 0 = unlimited lifetime deals
    deal_cool_down: float = 0.0  # seconds
    start_conditions: list[str] = field(default_factory=lambda: ["asap"])

    dca_stop_loss_enabled: bool = False
    dca_stop_loss_percent: float = 0.0
    dca_stop_loss_reference: str = "average"
    dca_stop_loss_move_breakeven: bool = False
    dca_stop_loss_breakeven_trigger: float = 0.0

    dca_trailing_stop_enabled: bool = False
    dca_trailing_stop_distance: float = 0.0
    dca_trailing_activate_profit: float = 0.0
    dca_trailing_replaces_take_profit: bool = True

    # Leverage: "same margin, N x bigger position" — first_order_amount/
    # dca_order_amount are MARGIN, actual notional = margin * leverage.
    # leverage=1 behaves like spot (no liquidation risk). See
    # exchange/bybit_client.py's ensure_leverage and dca_math's
    # calculate_liquidation_price for how this is enforced.
    leverage: float = 1.0

    # Position sizing: rescales first_order_amount/dca_order_amount
    # (preserving their ratio) at every deal start so the full ladder's
    # margin requirement equals funds_utilization_percent of the
    # exchange's CURRENT available balance — mirrors
    # strategy/backtest.py's identical mechanism.
    auto_size_to_funds: bool = True
    funds_utilization_percent: float = 98.0

    deal_count: int = 0


class DealStatus(IntEnum):
    ACTIVE = 0
    CLOSED = 1


@dataclass
class Deal:
    bot_id: str
    pair: str
    orders: list[OrderRung]
    deal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: DealStatus = DealStatus.ACTIVE
    is_start: int = 0  # 0 = base order not yet filled, 1 = monitoring/safety-order phase
    filled_count: int = 0

    paused: bool = False
    paused_buy: bool = False
    paused_sell: bool = False
    pause_reason: Optional[str] = None
    panic_sell: bool = False
    canceled: bool = False
    stop_loss: bool = False
    liquidated: bool = False
    stop_loss_breakeven_armed: bool = False
    active_stop_loss_price: float = 0.0
    trail_high_price: float = 0.0

    # Real funding cost accrued while this deal is open — mirrors
    # backtest.py's funding_events accounting so paper/live results are
    # economically comparable to a backtest of the same strategy, not
    # systematically more profitable for a reason unrelated to the
    # strategy itself. last_funding_settlement_ts is set to the base
    # order's fill time (see dca_bot.py's _handle_base_order) so funding
    # that happened before the deal opened is never charged.
    funding_cost_quote: float = 0.0
    last_funding_settlement_ts: float = 0.0

    config: Optional[BotConfig] = None  # The position’s own configuration, retained after close.
    sell_data: Optional[dict] = None
    date_opened: float = field(default_factory=time.time)
    date_closed: Optional[float] = None

    # Transient signal from the engine to DCABotManager, set only when
    # bot.reverse_drawdown_percent fired: the closing order ALSO opened
    # a new position in the opposite direction (see dca_bot.py's
    # _handle_sell), so the manager must register a monitoring engine
    # for it — not go through the normal fresh-order auto-chain path,
    # since the fill already happened. {"side": ..., "fill_price": ...,
    # "qty": ...}. Not meant to be read after _on_deal_complete consumes it.
    pending_flip: Optional[dict] = None
