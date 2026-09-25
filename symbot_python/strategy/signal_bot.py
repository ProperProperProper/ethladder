"""Start-condition helper.

The broader webhook/signal-ticker resolution, alert-body building, and
graceful-close guard that a full Signal-Bot/webhook feature would need
were never wired into anything here — that feature was out of scope
from the start — so only this one function is actually called (by
DCABotManager's auto-chain logic).
"""

from __future__ import annotations


def is_api_start(start_conditions: list[str]) -> bool:
    """True if this bot's primary start condition is 'api' (manual-webhook
    mode) rather than 'asap' or an external signal string (e.g.
    'signal|<source>|...'). Callers use this to decide whether a
    completed deal should ever auto-chain a replacement (an api-mode
    bot never does).
    """
    return bool(start_conditions) and start_conditions[0].strip().lower() == "api"
