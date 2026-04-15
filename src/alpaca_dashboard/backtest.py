"""Walk-forward backtester adapted from market_pulse/backend/backtest_algos.py.

Pipeline per (ticker, day, algo):
    1. Compute PGI from 60-day price momentum + RSI (simplified — no news).
    2. Compute 20-day annualised HV → sigma.
    3. Ask ``select_strategy_for_tier(S, sigma, expiry, dte, pgi, tier)`` for a
       strategy dict.
    4. Walk forward ``dte`` bars and score the outcome (win/loss/neutral).
    5. Persist the pulse row to SQLite (``store.save_pulse``) tagged with
       ``job_id``.

The coefficient-tuning layer on top: per-algo overrides fetched from
``store.get_coefficients(algo_id)`` scale position size, PGI entry threshold,
and target-DTE. Kept small so the user can tune from /admin sliders.
"""
from __future__ import annotations

import logging
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import json

from . import store
from .classify import build_regime_series, classify_cap, classify_regime
from .historical_data import fetch_history
from .strategies import select_strategy_for_tier

log = logging.getLogger(__name__)

ALGOS = ["degen", "surge", "moderate", "sentinel", "fortress"]

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "NVDA", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
    "AMD", "NFLX", "CRM", "JPM", "GS", "BAC",
    "XOM", "CVX", "NKE", "DIS", "BA", "HD",
    "UNH", "JNJ", "PFE", "ABBV", "MRK",
    "V", "MA", "PYPL",
]

DEFAULT_DTE_MAP = {"degen": 3, "surge": 10, "moderate": 21, "sentinel": 35, "fortress": 45}
DEFAULT_PGI_ENTRY = 0.0        # |pgi| >= this to take a signal
DEFAULT_SIZE_MULT = 1.0        # scales the strategy's nominal position sizing


# ── indicator helpers (copied from market_pulse, unchanged math) ──────────────

def _compute_rsi(closes: list[float], period: int = 14) -> list[float]:
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


def _compute_hv(closes: list[float], window: int = 20) -> float:
    if len(closes) < window + 1:
        return 0.22
    returns = [math.log(closes[i] / closes[i - 1])
               for i in range(len(closes) - window, len(closes))
               if closes[i - 1] > 0]
    if len(returns) < 5:
        return 0.22
    return statistics.stdev(returns) * math.sqrt(252)


def _json_safe(obj):
    """Best-effort conversion of strategy-dict contents (which may contain
    ``datetime.date`` or similar) into JSON-serialisable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def _compute_pgi(closes: list[float], idx: int) -> float:
    if idx < 50:
        return 0.0
    S = closes[idx]
    sma20 = sum(closes[idx - 19:idx + 1]) / 20
    sma50 = sum(closes[idx - 49:idx + 1]) / 50
    mom = (S - sma20) / sma20 if sma20 > 0 else 0
    mom50 = (S - sma50) / sma50 if sma50 > 0 else 0
    rsi_vals = _compute_rsi(closes[:idx + 1])
    rsi = rsi_vals[idx] if idx < len(rsi_vals) and not math.isnan(rsi_vals[idx]) else 50
    pgi = (mom * 200 * 0.4) + (mom50 * 200 * 0.3) + ((rsi - 50) / 50 * 100 * 0.3)
    return max(-100.0, min(100.0, round(pgi, 1)))


# ── walk-forward outcome evaluators (stock + option) ──────────────────────────

def _walk_forward_stock(entry, target, stop, highs, lows, max_bars=30):
    bars = min(len(highs), max_bars)
    for i in range(bars):
        if highs[i] >= target:
            return ("hit_target", "win", target, (target - entry) / entry * 100)
        if lows[i] <= stop:
            return ("stopped_out", "loss", stop, (stop - entry) / entry * 100)
    if bars > 0:
        final = (highs[-1] + lows[-1]) / 2
        pnl = (final - entry) / entry * 100
        return ("expired", "win" if pnl > 0 else "loss", final, pnl)
    return ("expired", "neutral", entry, 0.0)


def _walk_forward_option(entry_price, strategy, future_closes, max_bars=30):
    dte = strategy.get("dte", 14)
    bars = min(len(future_closes), max(dte, 1))
    if bars == 0:
        return ("expired", "neutral", entry_price, 0.0)
    P = future_closes[min(bars - 1, len(future_closes) - 1)]
    stype = strategy.get("strategy_type", "")
    credit = strategy.get("net_credit") or 0
    debit = strategy.get("net_debit") or 0

    if stype == "cash_secured_put":
        K = strategy.get("strike", entry_price * 0.93)
        if P >= K:
            return ("hit_target", "win", P, credit / K * 100 if K else 0)
        loss = K - P - credit
        return ("stopped_out", "loss" if loss > 0 else "win", P, -loss / K * 100 if K else 0)
    if stype == "iron_condor":
        put_K = strategy.get("strike") or strategy.get("short_put") or entry_price * 0.94
        call_K = strategy.get("strike_call_sell") or strategy.get("short_call") or entry_price * 1.06
        if put_K <= P <= call_K:
            return ("hit_target", "win", P, 100.0)
        return ("stopped_out", "loss", P, -100.0)
    if stype in ("bull_put_spread", "bull_call_spread"):
        K = strategy.get("strike", entry_price * 0.95)
        breakeven = K + debit if stype == "bull_call_spread" and debit else K
        if P >= breakeven:
            return ("hit_target", "win", P, min(credit / max(debit, 0.01) * 100 if debit else 100, 500))
        if debit and P >= K:
            return ("stopped_out", "loss", P, (P - K - debit) / max(debit, 0.01) * 100)
        return ("stopped_out", "loss", P, -100.0)
    if stype in ("bear_put_spread", "bear_call_spread"):
        K = strategy.get("strike", entry_price * 1.05)
        breakeven = K - debit if stype == "bear_put_spread" and debit else K
        if P <= breakeven:
            return ("hit_target", "win", P, min(credit / max(debit, 0.01) * 100 if debit else 100, 500))
        if debit and P <= K:
            return ("stopped_out", "loss", P, (K - P - debit) / max(debit, 0.01) * 100)
        return ("stopped_out", "loss", P, -100.0)
    if stype == "collar":
        put_K = strategy.get("strike") or strategy.get("put_strike") or entry_price * 0.95
        call_K = strategy.get("strike_call") or strategy.get("call_strike") or entry_price * 1.07
        total = (P - entry_price) + max(0, put_K - P) + -max(0, P - call_K) + credit - debit
        pct = total / entry_price * 100
        return ("hit_target" if total >= 0 else "stopped_out",
                "win" if total >= 0 else "loss", P, pct)
    if stype == "protective_put":
        K = strategy.get("strike", entry_price * 0.93)
        total = (P - entry_price) + max(0, K - P) - debit
        pct = total / entry_price * 100
        return ("hit_target" if total >= 0 else "stopped_out",
                "win" if total >= 0 else "loss", P, pct)
    if stype == "covered_call":
        K = strategy.get("strike", entry_price * 1.05)
        total = (min(P, K) - entry_price) + credit
        pct = total / entry_price * 100
        return ("hit_target" if total >= 0 else "stopped_out",
                "win" if total >= 0 else "loss", P, pct)
    if stype in ("long_call", "long_put", "long_straddle"):
        if stype == "long_call":
            K = strategy.get("strike", entry_price * 1.02)
            payoff = max(0, P - K) - debit
        elif stype == "long_put":
            K = strategy.get("strike", entry_price * 0.98)
            payoff = max(0, K - P) - debit
        else:
            K = strategy.get("strike", entry_price)
            payoff = max(0, P - K) + max(0, K - P) - debit
        pct = payoff / max(debit, 0.01) * 100 if debit else 0
        return ("hit_target" if payoff > 0 else "stopped_out",
                "win" if payoff > 0 else "loss", P, pct)
    return ("expired", "neutral", P, 0.0)


# ── public API ────────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    algo_id: str
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    days: int = 365
    target_pulses: int = 120
    job_id: str | None = None
    progress_cb: Callable[[int, int, str], None] | None = None   # (done, target, msg)
    stop_cb: Callable[[], bool] | None = None                    # returns True → abort


def run_single_algo(p: BacktestParams) -> dict:
    """Run a backtest for ONE algo. Returns a summary dict."""
    if p.algo_id not in ALGOS:
        raise ValueError(
            f"Algo '{p.algo_id}' is not backtestable. "
            f"Only ready algos are supported: {ALGOS}. "
            f"Planned algos (reddit_play, er_play, dividend_play) need a "
            f"strategy selector before they can run."
        )
    job_id = p.job_id or f"job_{uuid.uuid4().hex[:8]}"
    coefs = store.get_coefficients(p.algo_id)
    target_dte = int(coefs.get("target_dte", DEFAULT_DTE_MAP.get(p.algo_id, 21)))
    pgi_entry = float(coefs.get("pgi_entry", DEFAULT_PGI_ENTRY))
    size_mult = float(coefs.get("size_mult", DEFAULT_SIZE_MULT))  # noqa: F841 (metadata)

    # Fetch SPY once per run and precompute the regime table so tagging each
    # pulse is O(1). If SPY fetch fails the run still proceeds — we just tag
    # everything as "range" rather than block the backtest.
    regime_series: dict = {}
    spy_hist = fetch_history("SPY", p.days + 260)   # +260 for the 200-SMA window
    if spy_hist and spy_hist.get("close"):
        spy_closes = spy_hist["close"]
        spy_n = len(spy_closes)
        today = date.today()
        spy_dates = [today - timedelta(days=spy_n - 1 - i) for i in range(spy_n)]
        regime_series = build_regime_series(spy_closes, spy_dates)
    else:
        log.warning("SPY history fetch failed — pulses will be tagged regime='range'")

    signals: list[dict] = []
    aborted = False

    for ticker in p.tickers:
        if p.stop_cb and p.stop_cb():
            aborted = True
            break
        if len(signals) >= p.target_pulses:
            break
        if p.progress_cb:
            p.progress_cb(len(signals), p.target_pulses, f"fetching {ticker}")

        hist = fetch_history(ticker, p.days + 60)
        if not hist or not hist.get("close"):
            continue

        closes = hist["close"]
        highs = hist.get("high", closes)
        lows = hist.get("low", closes)
        n = len(closes)
        if n < 80:
            continue
        today = date.today()
        dates = [today - timedelta(days=n - 1 - i) for i in range(n)]

        # spread entries across the window rather than clustering
        skip_interval = max(1, (n - 75) // max(1, (p.target_pulses // max(1, len(p.tickers)) + 1)))

        for idx in range(60, n - 15):
            if p.stop_cb and p.stop_cb():
                aborted = True
                break
            if len(signals) >= p.target_pulses:
                break
            if (idx - 60) % skip_interval != 0:
                continue

            S = closes[idx]
            if S <= 0:
                continue
            pgi = _compute_pgi(closes, idx)
            if abs(pgi) < pgi_entry:
                continue
            sigma = max(_compute_hv(closes[:idx + 1]) * 1.15, 0.10)
            entry_date = dates[idx]

            expiry = entry_date + timedelta(days=target_dte)
            while expiry.weekday() != 4:   # nearest Friday
                expiry += timedelta(days=1)
            actual_dte = (expiry - entry_date).days

            try:
                strat = select_strategy_for_tier(
                    S, sigma, expiry, er_dte=actual_dte, pgi=pgi,
                    er_label="backtest", risk_tier=p.algo_id, ticker=ticker,
                )
            except Exception as e:
                log.debug(f"strategy select error {ticker} {entry_date}: {e}")
                continue
            if not strat:
                continue

            stype = strat.get("strategy_type", "")
            fut_c = closes[idx + 1: idx + 1 + actual_dte + 5]
            fut_h = highs[idx + 1: idx + 1 + actual_dte + 5]
            fut_l = lows[idx + 1: idx + 1 + actual_dte + 5]
            if not fut_c:
                continue

            if stype in ("Stock Trade", "stock_buy"):
                target = strat.get("target_price") or S * 1.05
                stop_px = strat.get("stop_price") or S * 0.97
                status, outcome, out_price, pnl_pct = _walk_forward_stock(
                    S, target, stop_px, fut_h, fut_l, max_bars=actual_dte,
                )
            else:
                status, outcome, out_price, pnl_pct = _walk_forward_option(
                    S, strat, fut_c, max_bars=actual_dte,
                )

            # Regime + cap tagging (look-ahead-free: regime_series is built
            # from SPY data; at entry_date we only use SPY up-to-and-including
            # that date so no information leak).
            regime = classify_regime(regime_series, entry_date)
            cap = classify_cap(ticker)

            # Snapshot the indicator values that drove this signal — lets us
            # later replay the same pulse against a different indicator blend
            # without re-running the backtest.
            rsi_series_up_to_now = _compute_rsi(closes[:idx + 1])
            rsi14_now = rsi_series_up_to_now[idx] if idx < len(rsi_series_up_to_now) else float("nan")
            sma20 = sum(closes[idx - 19:idx + 1]) / 20 if idx >= 19 else float("nan")
            sma50 = sum(closes[idx - 49:idx + 1]) / 50 if idx >= 49 else float("nan")
            indicators = {
                "pgi": pgi,
                "rsi14": None if math.isnan(rsi14_now) else round(rsi14_now, 2),
                "sma20": None if math.isnan(sma20) else round(sma20, 2),
                "sma50": None if math.isnan(sma50) else round(sma50, 2),
                "hv20": round(_compute_hv(closes[:idx + 1]), 4),
                "mom20": round((S - sma20) / sma20, 4) if sma20 and not math.isnan(sma20) else None,
                "mom50": round((S - sma50) / sma50, 4) if sma50 and not math.isnan(sma50) else None,
            }

            pulse_id = f"bt_{p.algo_id}_{ticker}_{entry_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            row = {
                "pulse_id": pulse_id,
                "algo_id": p.algo_id,
                "ticker": ticker,
                "pulse_type": stype,
                "strategy_label": strat.get("strategy_label", stype),
                "entry_date": entry_date.isoformat(),
                "entry_price": round(S, 2),
                "expiry": expiry.isoformat(),
                "dte": actual_dte,
                "pgi": pgi,
                "sigma": round(sigma, 3),
                "status": status,
                "outcome": outcome,
                "outcome_price": round(out_price, 2),
                "outcome_pnl_pct": round(pnl_pct, 2),
                "selection_reason": (strat.get("selection_reason") or "")[:200],
                "job_id": job_id,
                "top_rec_json": json.dumps(_json_safe(strat)),
                "indicators_json": json.dumps(indicators),
                "market_regime": regime,
                "cap_bucket": cap,
            }
            store.save_pulse(row)
            signals.append(row)

            if p.progress_cb and len(signals) % 10 == 0:
                p.progress_cb(len(signals), p.target_pulses, f"{ticker} {entry_date}")

    return _summarise(p.algo_id, signals, aborted)


def _summarise(algo_id: str, signals: list[dict], aborted: bool) -> dict:
    total = len(signals)
    if total == 0:
        return {"algo_id": algo_id, "total": 0, "win_rate": 0, "avg_pnl": 0, "aborted": aborted}
    wins = sum(1 for s in signals if s["outcome"] == "win")
    losses = sum(1 for s in signals if s["outcome"] == "loss")
    pnl = [s["outcome_pnl_pct"] for s in signals]
    win_pnl = [s["outcome_pnl_pct"] for s in signals if s["outcome"] == "win"]
    loss_pnl = [s["outcome_pnl_pct"] for s in signals if s["outcome"] == "loss"]
    pf = (abs(sum(win_pnl)) / abs(sum(loss_pnl))) if loss_pnl and sum(loss_pnl) != 0 else None
    return {
        "algo_id": algo_id,
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1),
        "avg_pnl": round(sum(pnl) / total, 2),
        "avg_win": round(sum(win_pnl) / len(win_pnl), 2) if win_pnl else 0,
        "avg_loss": round(sum(loss_pnl) / len(loss_pnl), 2) if loss_pnl else 0,
        "total_pnl": round(sum(pnl), 1),
        "profit_factor": round(pf, 2) if pf is not None else None,
        "aborted": aborted,
    }
