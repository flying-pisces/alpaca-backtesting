"""Alpaca historical bars adapter.

The market_pulse backtester expects a ``fetch_history(symbol, days)`` callable
returning a dict of ``{close, high, low, open, volume}`` arrays. We use
``alpaca-py``'s ``StockHistoricalDataClient`` against the free IEX feed.
Any configured account's key works since market data is shared.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from functools import cache
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

log = logging.getLogger(__name__)


def _first_configured_key() -> tuple[str, str] | None:
    for prefix in ("DEGEN", "SURGE", "MODERATE", "SENTINEL", "FORTRESS"):
        k = os.getenv(f"ALPACA_{prefix}_KEY")
        s = os.getenv(f"ALPACA_{prefix}_SECRET")
        if k and s:
            return k, s
    return None


@cache
def _client() -> StockHistoricalDataClient | None:
    creds = _first_configured_key()
    if not creds:
        log.warning("No Alpaca credentials found — cannot fetch historical bars")
        return None
    return StockHistoricalDataClient(api_key=creds[0], secret_key=creds[1])


def fetch_history(symbol: str, days: int = 180) -> Optional[dict]:
    c = _client()
    if c is None:
        return None
    end = datetime.now(timezone.utc)
    # Pad for weekends/holidays; SIP end has a 15-min delay on free, use day-old cutoff
    start = end - timedelta(days=days + 40)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end - timedelta(minutes=20),
            feed="iex",
            limit=days + 40,
        )
        resp = c.get_stock_bars(req)
    except Exception as e:
        log.warning(f"Alpaca bars fetch failed for {symbol}: {e}")
        return None

    bars = resp.data.get(symbol, []) if hasattr(resp, "data") else []
    if not bars:
        return None
    return {
        "close": [float(b.close) for b in bars],
        "high": [float(b.high) for b in bars],
        "low": [float(b.low) for b in bars],
        "open": [float(b.open) for b in bars],
        "volume": [float(b.volume) for b in bars],
    }
