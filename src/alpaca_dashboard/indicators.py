"""Shared indicator + util helpers used by both the backtest and the live
engine. Split into its own module so neither module has to reach into the
other's private helpers (and so that Streamlit Cloud's module cache never
ends up holding a partial copy without the name we need).
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime


def compute_rsi(closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    result = [float("nan")] * n
    if n < period + 1:
        return result
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    result[period] = 100 - 100 / (1 + avg_g / avg_l) if avg_l else 100.0
    for i in range(period + 1, n):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0)) / period
        result[i] = 100 - 100 / (1 + avg_g / avg_l) if avg_l else 100.0
    return result


def compute_hv(closes: list[float], window: int = 20) -> float:
    if len(closes) < window + 1:
        return 0.22
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - window, len(closes))
        if closes[i - 1] > 0
    ]
    if len(returns) < 5:
        return 0.22
    return statistics.stdev(returns) * math.sqrt(252)


def compute_pgi(closes: list[float], idx: int) -> float:
    if idx < 50:
        return 0.0
    S = closes[idx]
    sma20 = sum(closes[idx - 19:idx + 1]) / 20
    sma50 = sum(closes[idx - 49:idx + 1]) / 50
    mom = (S - sma20) / sma20 if sma20 > 0 else 0
    mom50 = (S - sma50) / sma50 if sma50 > 0 else 0
    rsi_vals = compute_rsi(closes[:idx + 1])
    rsi = rsi_vals[idx] if idx < len(rsi_vals) and not math.isnan(rsi_vals[idx]) else 50
    pgi = (mom * 200 * 0.4) + (mom50 * 200 * 0.3) + ((rsi - 50) / 50 * 100 * 0.3)
    return max(-100.0, min(100.0, round(pgi, 1)))


def json_safe(obj):
    """Best-effort conversion of nested structures into JSON-serialisable
    primitives. ``datetime.date`` / ``datetime`` become iso strings; anything
    unknown falls back to ``str``."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    return str(obj)
