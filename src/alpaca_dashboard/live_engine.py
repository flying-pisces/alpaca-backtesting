"""Go Live Pulse engine — background thread that scans the universe on a
schedule and emits live pulses matching market_pulse's schema.

Design:
  * One singleton daemon thread (survives Streamlit reruns, dies with the
    process).
  * Each cycle walks the universe once. For each ticker, bars are fetched
    through ``data_cache.fetch_bars`` (TTL) so five algos share one API
    call per ticker per cycle.
  * For each (ticker, ready_algo) pair, the live PGI/σ is computed at
    the most recent bar; if ``select_strategy_for_tier`` returns a
    strategy, a live pulse is written.
  * Dedup: one pulse per (algo, ticker) per ``dedup_window_sec`` (default
    30 min). A new pulse only fires if the PGI has moved > ``pgi_delta``
    points OR the strategy_type changed.
  * All pulses tagged ``generated_by = "go_live_{algo_id}"`` for easy
    filtering in the dashboard and on market_pulse's side.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from . import store
from .backtest import ALGOS, DEFAULT_DTE_MAP, DEFAULT_TICKERS
from .classify import build_regime_series, classify_cap, classify_regime
from .data_cache import BARS_CACHE, fetch_bars
from .indicators import compute_hv, compute_pgi, compute_rsi, json_safe
from .strategies import select_strategy_for_tier

log = logging.getLogger(__name__)


# ── Config defaults ─────────────────────────────────────────────────────────

DEFAULT_CYCLE_SEC = 300       # 5 min between full-universe scans
DEFAULT_DEDUP_SEC = 1800      # 30 min
DEFAULT_PGI_DELTA = 10.0      # re-fire threshold: PGI changed > this
HISTORY_DAYS = 400            # enough for 200-SMA regime classifier


# ── State ───────────────────────────────────────────────────────────────────

@dataclass
class EngineState:
    running: bool = False
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_cycle_generated: int = 0
    last_cycle_skipped_dedup: int = 0
    last_cycle_errors: int = 0
    total_generated: int = 0
    total_cycles: int = 0
    last_error: Optional[str] = None
    # Config (mirrored so the UI can read it)
    cycle_sec: int = DEFAULT_CYCLE_SEC
    dedup_sec: int = DEFAULT_DEDUP_SEC
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    algos: list[str] = field(default_factory=lambda: list(ALGOS))


_STATE = EngineState()
_STATE_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()

# In-memory dedup cache: {(algo, ticker): (timestamp_iso, pgi, strategy_type)}.
# Small and ephemeral — if the process restarts, the worst case is a
# duplicate pulse in the first cycle after restart, which INSERT OR IGNORE
# (via pulse_id uniqueness) prevents from actually duplicating downstream.
_DEDUP_STATE: dict[tuple[str, str], tuple[float, float, str]] = {}


def state_snapshot() -> dict:
    with _STATE_LOCK:
        return {
            "running": _STATE.running,
            "started_at": _STATE.started_at,
            "stopped_at": _STATE.stopped_at,
            "last_cycle_at": _STATE.last_cycle_at,
            "last_cycle_generated": _STATE.last_cycle_generated,
            "last_cycle_skipped_dedup": _STATE.last_cycle_skipped_dedup,
            "last_cycle_errors": _STATE.last_cycle_errors,
            "total_generated": _STATE.total_generated,
            "total_cycles": _STATE.total_cycles,
            "last_error": _STATE.last_error,
            "cycle_sec": _STATE.cycle_sec,
            "dedup_sec": _STATE.dedup_sec,
            "tickers": list(_STATE.tickers),
            "algos": list(_STATE.algos),
            "cache": BARS_CACHE.stats(),
        }


# ── Core scan logic ─────────────────────────────────────────────────────────

def _compute_live_indicators(closes: list[float], idx: int) -> dict:
    """Same blend as backtest._json_safe-adjacent indicators, sampled at
    the latest bar so the live pulse carries the values the signal used."""
    pgi = compute_pgi(closes, idx)
    hv = compute_hv(closes[:idx + 1])
    rsi_series = compute_rsi(closes[:idx + 1])
    rsi14 = rsi_series[idx] if idx < len(rsi_series) else float("nan")
    sma20 = sum(closes[idx - 19:idx + 1]) / 20 if idx >= 19 else float("nan")
    sma50 = sum(closes[idx - 49:idx + 1]) / 50 if idx >= 49 else float("nan")
    S = closes[idx]
    return {
        "pgi": pgi,
        "rsi14": None if math.isnan(rsi14) else round(rsi14, 2),
        "sma20": None if math.isnan(sma20) else round(sma20, 2),
        "sma50": None if math.isnan(sma50) else round(sma50, 2),
        "hv20": round(hv, 4),
        "mom20": round((S - sma20) / sma20, 4) if sma20 and not math.isnan(sma20) else None,
        "mom50": round((S - sma50) / sma50, 4) if sma50 and not math.isnan(sma50) else None,
    }


def _should_skip_dedup(algo: str, ticker: str, now: float, pgi: float,
                       strategy_type: str, dedup_sec: int,
                       pgi_delta: float = DEFAULT_PGI_DELTA) -> bool:
    key = (algo, ticker)
    prev = _DEDUP_STATE.get(key)
    if not prev:
        return False
    prev_ts, prev_pgi, prev_stype = prev
    if now - prev_ts >= dedup_sec:
        return False
    if abs(pgi - prev_pgi) > pgi_delta:
        return False
    if strategy_type and strategy_type != prev_stype:
        return False
    return True


def _emit_pulse(algo: str, ticker: str, S: float, sigma: float, pgi: float,
                strat: dict, indicators: dict, regime: str, cap: str,
                generated_at_iso: str, entry_date: date) -> None:
    stype = strat.get("strategy_type", "") or ""
    stamp = datetime.strptime(generated_at_iso, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d%H%M%S")
    pulse_id = f"live_{algo}_{ticker}_{stamp}_{uuid.uuid4().hex[:4]}"

    # For live pulses the outcome isn't known — status=active, outcome=None.
    # The converter reads these fields; market_pulse's outcome_tracker will
    # resolve them live, or our own backtest replay will later.
    dte = int(strat.get("dte") or DEFAULT_DTE_MAP.get(algo, 21))
    expiry = entry_date + timedelta(days=dte)

    row = {
        "pulse_id": pulse_id,
        "algo_id": algo,
        "ticker": ticker,
        "pulse_type": stype,
        "strategy_label": strat.get("strategy_label", stype),
        "entry_date": entry_date.isoformat(),
        "entry_price": round(S, 2),
        "expiry": expiry.isoformat(),
        "dte": dte,
        "pgi": pgi,
        "sigma": round(sigma, 3),
        "status": "active",
        "outcome": None,
        "outcome_price": None,
        "outcome_pnl_pct": None,
        "selection_reason": (strat.get("selection_reason") or "")[:200],
        "job_id": f"go_live_{algo}",
        "top_rec_json": json.dumps(json_safe(strat)),
        "indicators_json": json.dumps(indicators),
        "market_regime": regime,
        "cap_bucket": cap,
        "generated_at": generated_at_iso,
    }
    store.save_pulse(row)


def _run_cycle() -> dict:
    """One pass over the universe. Returns per-cycle counters."""
    generated = 0
    skipped = 0
    errors = 0

    with _STATE_LOCK:
        tickers = list(_STATE.tickers)
        algos = [a for a in _STATE.algos if a in ALGOS]
        dedup_sec = _STATE.dedup_sec

    # Prebuild the regime lookup once per cycle (one SPY fetch goes through
    # the cache — next ticker's fetch avoids repeat work).
    regime_series: dict = {}
    spy_bars = fetch_bars("SPY", HISTORY_DAYS)
    if spy_bars and spy_bars.get("close"):
        spy_closes = spy_bars["close"]
        spy_n = len(spy_closes)
        today = date.today()
        spy_dates = [today - timedelta(days=spy_n - 1 - i) for i in range(spy_n)]
        regime_series = build_regime_series(spy_closes, spy_dates)

    now_unix = time.time()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = date.today()

    for ticker in tickers:
        if _STOP_EVENT.is_set():
            break
        try:
            hist = fetch_bars(ticker, HISTORY_DAYS)
            if not hist or not hist.get("close"):
                continue
            closes = hist["close"]
            if len(closes) < 80:
                continue
            idx = len(closes) - 1   # most recent bar
            S = closes[idx]
            if S <= 0:
                continue

            indicators = _compute_live_indicators(closes, idx)
            pgi = indicators["pgi"]
            sigma = max(compute_hv(closes[:idx + 1]) * 1.15, 0.10)
            regime = classify_regime(regime_series, today)
            cap = classify_cap(ticker)

            for algo in algos:
                coefs = store.get_coefficients(algo)
                pgi_entry = float(coefs.get("pgi_entry", 0.0))
                if abs(pgi) < pgi_entry:
                    continue
                target_dte = int(coefs.get("target_dte", DEFAULT_DTE_MAP.get(algo, 21)))
                expiry = today + timedelta(days=target_dte)
                while expiry.weekday() != 4:
                    expiry += timedelta(days=1)
                actual_dte = (expiry - today).days

                try:
                    strat = select_strategy_for_tier(
                        S, sigma, expiry, er_dte=actual_dte, pgi=pgi,
                        er_label="live", risk_tier=algo, ticker=ticker,
                    )
                except Exception as e:   # noqa: BLE001
                    errors += 1
                    log.debug(f"select_strategy_for_tier error {algo}/{ticker}: {e}")
                    continue
                if not strat:
                    continue

                stype = strat.get("strategy_type", "") or ""
                if _should_skip_dedup(algo, ticker, now_unix, pgi, stype, dedup_sec):
                    skipped += 1
                    continue

                try:
                    _emit_pulse(
                        algo, ticker, S, sigma, pgi, strat, indicators,
                        regime, cap, now_iso, today,
                    )
                    _DEDUP_STATE[(algo, ticker)] = (now_unix, pgi, stype)
                    generated += 1
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    log.warning(f"emit_pulse failed {algo}/{ticker}: {e}")
        except Exception as e:   # noqa: BLE001
            errors += 1
            log.warning(f"ticker {ticker} scan crashed: {e}")

    return {"generated": generated, "skipped": skipped, "errors": errors}


# ── Lifecycle ───────────────────────────────────────────────────────────────

_LAST_PUSH_WRITTEN = 0


def _auto_push() -> None:
    """If ``AUTO_PUSH_SQLITE_PATH`` env is set, push new pulses to
    market_pulse's DB after each cycle.

    During market hours the DB is typically locked by market_pulse's server.
    Failures are silently retried on the next cycle — the cursor ensures no
    rows are missed. A successful push is logged at INFO level so the user
    can see when it first gets through (usually right after market close).
    """
    global _LAST_PUSH_WRITTEN
    from .ingestion import HttpDestination, SqliteDestination, push

    destinations: list = []

    # Prefer HTTP if INGEST_API_KEY is set (bypasses SQLite lock entirely).
    ingest_key = os.getenv("INGEST_API_KEY", "").strip()
    if ingest_key:
        destinations.append(HttpDestination(auth_token=ingest_key))

    # SQLite fallback / additional destination.
    sqlite_path = os.getenv("AUTO_PUSH_SQLITE_PATH", "").strip()
    if sqlite_path:
        destinations.append(SqliteDestination(sqlite_path))

    for dest in destinations:
        try:
            ok, msg = dest.healthcheck()
            if not ok:
                continue
            result = push(dest, batch_size=500)
            if result.total_written:
                _LAST_PUSH_WRITTEN += result.total_written
                log.info(f"auto-push: +{result.total_written} rows → {dest.name} "
                         f"(cumulative: {_LAST_PUSH_WRITTEN})")
        except Exception:  # noqa: BLE001
            pass   # transient error — cursor persists, next cycle retries


def _loop() -> None:
    """Daemon body. Stops when ``_STOP_EVENT`` is set.

    Market-hours gate: if ``is_market_open()`` returns ``False``, the loop
    sleeps without scanning. This means the engine can be "always on" — it
    automatically wakes when the market opens and idles off-hours.
    """
    from .market_clock import is_market_open

    log.info("live engine started")
    while not _STOP_EVENT.is_set():
        # Gate: skip the scan when the market is closed.
        if not is_market_open():
            with _STATE_LOCK:
                _STATE.last_error = None
            # Sleep in ticks — check every 30s whether market opened.
            for _ in range(30):
                if _STOP_EVENT.is_set():
                    break
                time.sleep(1)
            continue

        t0 = time.time()
        try:
            counters = _run_cycle()
        except Exception as e:   # noqa: BLE001
            log.exception("live engine cycle crashed")
            counters = {"generated": 0, "skipped": 0, "errors": 1}
            with _STATE_LOCK:
                _STATE.last_error = f"{type(e).__name__}: {e}"
        with _STATE_LOCK:
            _STATE.last_cycle_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _STATE.last_cycle_generated = counters["generated"]
            _STATE.last_cycle_skipped_dedup = counters["skipped"]
            _STATE.last_cycle_errors = counters["errors"]
            _STATE.total_generated += counters["generated"]
            _STATE.total_cycles += 1
            cycle_sec = _STATE.cycle_sec

        # Auto-push to market_pulse after each successful cycle.
        if counters["generated"] > 0:
            _auto_push()

        # Wait in small ticks so stop() is responsive even mid-cycle.
        elapsed = time.time() - t0
        remaining = max(1.0, cycle_sec - elapsed)
        deadline = time.time() + remaining
        while time.time() < deadline and not _STOP_EVENT.is_set():
            time.sleep(min(1.0, deadline - time.time()))
    log.info("live engine stopped")


def start(
    tickers: Optional[list[str]] = None,
    algos: Optional[list[str]] = None,
    cycle_sec: Optional[int] = None,
    dedup_sec: Optional[int] = None,
) -> dict:
    """Idempotent — returns current state whether we started a new thread or
    the engine was already running."""
    global _THREAD
    with _STATE_LOCK:
        if _STATE.running and _THREAD and _THREAD.is_alive():
            return state_snapshot()
        if tickers:
            _STATE.tickers = [t.upper() for t in tickers if t]
        if algos:
            _STATE.algos = [a for a in algos if a in ALGOS]
        if cycle_sec:
            _STATE.cycle_sec = max(30, int(cycle_sec))
        if dedup_sec:
            _STATE.dedup_sec = max(60, int(dedup_sec))
        _STATE.running = True
        _STATE.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _STATE.stopped_at = None
        _STATE.last_error = None
    _STOP_EVENT.clear()
    _THREAD = threading.Thread(target=_loop, name="live_engine", daemon=True)
    _THREAD.start()
    return state_snapshot()


def stop() -> dict:
    with _STATE_LOCK:
        _STATE.running = False
        _STATE.stopped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _STOP_EVENT.set()
    # Don't join the thread — Streamlit request threads should return
    # quickly; the daemon thread exits on its own tick.
    return state_snapshot()


def run_once() -> dict:
    """Run a single cycle synchronously. Useful for tests and the
    "Run cycle now" button on /Admin."""
    counters = _run_cycle()
    with _STATE_LOCK:
        _STATE.last_cycle_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _STATE.last_cycle_generated = counters["generated"]
        _STATE.last_cycle_skipped_dedup = counters["skipped"]
        _STATE.last_cycle_errors = counters["errors"]
        _STATE.total_generated += counters["generated"]
        _STATE.total_cycles += 1
    return {**counters, **state_snapshot()}
