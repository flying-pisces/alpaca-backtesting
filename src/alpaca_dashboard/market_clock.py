"""Market clock — thin wrapper around Alpaca's ``get_clock()`` API.

Returns whether US equities markets are currently open. Used by:
  * ``live_engine._loop()`` to skip cycles off-hours
  * ``streamlit_app.py`` to auto-start the engine on boot when market is live
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import cache

from alpaca.trading.client import TradingClient

log = logging.getLogger(__name__)


def _first_configured_client() -> TradingClient | None:
    for prefix in ("DEGEN", "SURGE", "MODERATE", "SENTINEL", "FORTRESS",
                   "REDDIT_PLAY", "ER_PLAY", "DIVIDEND_PLAY"):
        k = os.getenv(f"ALPACA_{prefix}_KEY")
        s = os.getenv(f"ALPACA_{prefix}_SECRET")
        if k and s:
            return TradingClient(api_key=k, secret_key=s, paper=True)
    return None


def get_clock() -> dict:
    """Return ``{is_open, next_open, next_close}`` or a safe default.

    If every account key is misconfigured, returns ``is_open=False`` so the
    engine stays dormant rather than crashing.
    """
    client = _first_configured_client()
    if not client:
        log.warning("no configured Alpaca account — assuming market closed")
        return {"is_open": False, "next_open": None, "next_close": None}
    try:
        c = client.get_clock()
        return {
            "is_open": c.is_open,
            "next_open": c.next_open.isoformat() if c.next_open else None,
            "next_close": c.next_close.isoformat() if c.next_close else None,
            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
        }
    except Exception as e:  # noqa: BLE001
        log.warning(f"get_clock() failed: {e} — assuming closed")
        return {"is_open": False, "next_open": None, "next_close": None}


def is_market_open() -> bool:
    return get_clock().get("is_open", False)
