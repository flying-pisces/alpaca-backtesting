"""Regime + cap-bucket classifiers for backtest pulses.

Goal: tag each generated pulse with coarse market-context labels so we
can later segment backtest results ("does PGI work in bull markets?
what about small-caps?"). Keeping both classifiers stateless and fast —
they run on every pulse save.
"""
from __future__ import annotations

import math
from datetime import date
from functools import cache
from pathlib import Path

import yaml

from .settings import ROOT


# ── Market regime ────────────────────────────────────────────────────────────
#
# Rule-based from SPY price + 50/200-day SMAs. No lookahead bias — at
# ``entry_date`` we only use SPY data up to and including that date.

def build_regime_series(spy_closes: list[float], dates: list[date]) -> dict[date, str]:
    """Precompute a {date → regime} table from an SPY close-price series.

    Call once per backtest run (SPY is fetched once regardless of how many
    tickers we're scanning), then look up each entry_date in O(1).

    Regime rules:
      - ``bull``  — price > SMA200 AND SMA50 > SMA200
      - ``bear``  — price < SMA200 AND SMA50 < SMA200
      - ``range`` — everything in between (or if we don't have enough history)
    """
    out: dict[date, str] = {}
    n = len(spy_closes)
    if n == 0 or len(dates) != n:
        return out

    for i in range(n):
        if i < 199:
            out[dates[i]] = "range"
            continue
        price = spy_closes[i]
        sma50 = sum(spy_closes[i - 49:i + 1]) / 50
        sma200 = sum(spy_closes[i - 199:i + 1]) / 200
        if math.isnan(price) or math.isnan(sma50) or math.isnan(sma200):
            out[dates[i]] = "range"
            continue
        if price > sma200 and sma50 > sma200:
            out[dates[i]] = "bull"
        elif price < sma200 and sma50 < sma200:
            out[dates[i]] = "bear"
        else:
            out[dates[i]] = "range"
    return out


def classify_regime(regime_series: dict[date, str], entry_date: date) -> str:
    """Lookup with fallback to ``range`` if date missing (e.g., first 200 bars)."""
    return regime_series.get(entry_date, "range")


# ── Cap bucket ──────────────────────────────────────────────────────────────

@cache
def _cap_index() -> dict[str, str]:
    """Load config/cap_buckets.yaml once and invert it to ``{ticker: bucket}``."""
    p = ROOT / "config" / "cap_buckets.yaml"
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    idx: dict[str, str] = {}
    for bucket in ("large", "mid", "small"):
        for t in (raw.get(bucket) or []):
            idx[t.upper()] = bucket
    return idx


def classify_cap(ticker: str) -> str:
    """Return ``large | mid | small``. Defaults to ``large`` for unknown tickers
    because our default universe is mega-cap; override by listing explicitly in
    ``config/cap_buckets.yaml``.
    """
    return _cap_index().get((ticker or "").upper(), "large")
