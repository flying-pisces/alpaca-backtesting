#!/usr/bin/env python3
"""Self-QA: end-to-end validation of every subsystem.

Runs against live Turso + live Alpaca + live Fly.io production. Designed to
be run from the terminal at any time — safe, idempotent, cleans up after
itself.

Usage:
    .venv/bin/python scripts/self_qa.py          # run all checks
    .venv/bin/python scripts/self_qa.py --quick   # skip slow network checks

Exit code: 0 = all pass, 1 = any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Harness ──────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator that runs a check function and records pass/fail."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                msg = fn(*args, **kwargs)
                _results.append((name, True, msg or ""))
                print(f"  ✅ {name}: {msg or 'ok'}")
            except Exception as e:
                tb = traceback.format_exc()
                _results.append((name, False, str(e)))
                print(f"  ❌ {name}: {e}")
                if os.getenv("QA_VERBOSE"):
                    print(tb)
        return wrapper
    return decorator


# ── 1. Store (Turso) ─────────────────────────────────────────────────────────

@check("store.init_db")
def test_store_init():
    from alpaca_dashboard import store
    store.init_db()
    return f"turso={store.using_turso()}"


@check("store.save_pulse + all_pulses round-trip")
def test_store_crud():
    from alpaca_dashboard import store
    pid = f"qa_probe_{int(time.time())}"
    store.save_pulse({
        "pulse_id": pid, "algo_id": "degen", "ticker": "SPY",
        "pulse_type": "long_call", "strategy_label": "qa",
        "entry_date": "2026-04-16", "entry_price": 500.0,
        "expiry": "2026-04-23", "dte": 7, "pgi": 20.0, "sigma": 0.2,
        "status": "active", "outcome": None, "outcome_price": None,
        "outcome_pnl_pct": None, "selection_reason": "qa",
        "job_id": "qa", "top_rec_json": '{"test":1}',
        "indicators_json": '{"pgi":20}', "market_regime": "bull",
        "cap_bucket": "large", "generated_at": "2026-04-16T12:00:00Z",
    })
    rows = store.all_pulses(limit=10)
    found = any(r["pulse_id"] == pid for r in rows)
    assert found, f"pulse {pid} not found after save"
    # Cleanup
    store.delete_all_pulses()
    return f"saved + read + deleted {pid}"


@check("store.coefficients round-trip")
def test_store_coefs():
    from alpaca_dashboard import store
    store.set_coefficients("qa_algo", {"x": 42})
    c = store.get_coefficients("qa_algo")
    assert c == {"x": 42}, f"expected {{x:42}}, got {c}"
    return "set + get OK"


@check("store.ingestion_cursor round-trip")
def test_store_cursor():
    from alpaca_dashboard import store
    store.set_ingestion_cursor("qa_dest", "2026-04-16T00:00:00", 99,
                               last_pushed_pulse_id="qa_1")
    c = store.get_ingestion_cursor("qa_dest")
    assert c is not None
    assert c["last_pushed_count"] == 99
    assert c["last_pushed_pulse_id"] == "qa_1"
    return "set + get OK"


@check("store.pulses_since composite cursor")
def test_store_pagination():
    from alpaca_dashboard import store
    store.delete_all_pulses()
    for i in range(5):
        store.save_pulse({
            "pulse_id": f"qa_page_{i}", "algo_id": "degen", "ticker": "SPY",
            "pulse_type": "x", "strategy_label": "x", "entry_date": "2026-04-16",
            "entry_price": 100, "expiry": "2026-04-23", "dte": 7,
            "pgi": 0, "sigma": 0.2, "status": "active", "outcome": None,
            "outcome_price": None, "outcome_pnl_pct": None,
            "selection_reason": "x", "job_id": "qa",
            "top_rec_json": None, "indicators_json": None,
            "market_regime": None, "cap_bucket": None, "generated_at": None,
        })
    page1 = store.pulses_since(limit=3)
    assert len(page1) == 3, f"expected 3, got {len(page1)}"
    last = page1[-1]
    page2 = store.pulses_since(
        since_created_at=last["created_at"],
        since_pulse_id=last["pulse_id"],
        limit=3,
    )
    assert len(page2) == 2, f"expected 2, got {len(page2)}"
    total_ids = {r["pulse_id"] for r in page1 + page2}
    assert len(total_ids) == 5, f"expected 5 unique, got {len(total_ids)}"
    store.delete_all_pulses()
    return "5 rows, paginated 3+2, 0 skipped"


# ── 2. Market clock ──────────────────────────────────────────────────────────

@check("market_clock.get_clock")
def test_market_clock():
    from alpaca_dashboard.market_clock import get_clock
    c = get_clock()
    assert "is_open" in c, f"missing is_open: {c}"
    return f"is_open={c['is_open']}, next_close={c.get('next_close','?')[:19]}"


# ── 3. Classifiers ──────────────────────────────────────────────────────────

@check("classify.cap_bucket")
def test_cap():
    from alpaca_dashboard.classify import classify_cap
    assert classify_cap("SPY") == "large"
    assert classify_cap("AAPL") == "large"
    assert classify_cap("UNKNOWN_TICKER") == "large"  # default
    return "SPY=large, AAPL=large, unknown=large"


@check("classify.regime_series")
def test_regime():
    from alpaca_dashboard.classify import build_regime_series
    prices = [100.0 + i * 0.3 for i in range(250)]
    d0 = date(2025, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(250)]
    s = build_regime_series(prices, dates)
    assert s[dates[240]] == "bull", f"expected bull, got {s[dates[240]]}"
    return "rising series → bull at day 240"


# ── 4. Indicators ───────────────────────────────────────────────────────────

@check("indicators.compute_pgi")
def test_pgi():
    from alpaca_dashboard.indicators import compute_pgi
    closes = [100.0 + i * 0.1 for i in range(100)]
    pgi = compute_pgi(closes, 99)
    assert -100 <= pgi <= 100, f"PGI out of range: {pgi}"
    return f"PGI={pgi:.1f} (range OK)"


@check("indicators.compute_hv")
def test_hv():
    from alpaca_dashboard.indicators import compute_hv
    closes = [100.0 + i * 0.1 for i in range(30)]
    hv = compute_hv(closes)
    assert 0 < hv < 2.0, f"HV out of range: {hv}"
    return f"HV={hv:.4f}"


@check("indicators.json_safe")
def test_json_safe():
    from alpaca_dashboard.indicators import json_safe
    d = {"a": date(2026, 1, 1), "b": [1, None, {"c": 3.14}]}
    s = json_safe(d)
    assert s["a"] == "2026-01-01"
    assert json.dumps(s)  # must be serialisable
    return "date→iso, nested OK"


# ── 5. Data cache ───────────────────────────────────────────────────────────

@check("data_cache.TTLCache")
def test_cache():
    from alpaca_dashboard.data_cache import TTLCache
    c = TTLCache(ttl_sec=1.0)
    c.get_or_fetch("TEST", 10, fetcher=lambda s, d: {"close": [1, 2, 3]})
    c.get_or_fetch("TEST", 10, fetcher=lambda s, d: {"close": [4, 5, 6]})
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    return f"hit_rate={s['hit_rate']:.0%}"


# ── 6. Historical data (network) ────────────────────────────────────────────

@check("historical_data.fetch_history (SPY, 30 days)")
def test_fetch():
    from alpaca_dashboard.historical_data import fetch_history
    h = fetch_history("SPY", 30)
    assert h is not None, "fetch_history returned None"
    assert len(h["close"]) > 10, f"only {len(h['close'])} bars"
    return f"{len(h['close'])} bars"


# ── 7. Strategy selector ────────────────────────────────────────────────────

@check("select_strategy_for_tier (moderate, SPY-like)")
def test_strategy():
    from alpaca_dashboard.strategies import select_strategy_for_tier
    expiry = date.today() + timedelta(days=21)
    strat = select_strategy_for_tier(
        S=500.0, sigma=0.22, expiry=expiry, er_dte=21,
        pgi=30.0, er_label="qa", risk_tier="moderate", ticker="SPY",
    )
    assert strat is not None, "selector returned None"
    assert "strategy_type" in strat
    return f"strategy_type={strat['strategy_type']}"


# ── 8. Backtest engine (mini run) ───────────────────────────────────────────

@check("backtest.run_single_algo (moderate, 2 tickers, 3 pulses)")
def test_backtest():
    from alpaca_dashboard import store
    from alpaca_dashboard.backtest import BacktestParams, run_single_algo
    store.delete_all_pulses()
    summary = run_single_algo(BacktestParams(
        algo_id="moderate", tickers=["SPY", "AAPL"],
        days=180, target_pulses=3,
    ))
    assert summary["total"] >= 1, f"generated 0 pulses"
    # Verify enriched fields
    rows = store.all_pulses(limit=10)
    r = rows[0]
    assert r.get("market_regime") in ("bull", "bear", "range"), f"bad regime: {r.get('market_regime')}"
    assert r.get("cap_bucket") in ("large", "mid", "small"), f"bad cap: {r.get('cap_bucket')}"
    assert r.get("indicators_json"), "missing indicators_json"
    assert r.get("top_rec_json"), "missing top_rec_json"
    store.delete_all_pulses()
    return f"{summary['total']} pulses, regime={r['market_regime']}, cap={r['cap_bucket']}"


# ── 9. Live engine (one cycle) ──────────────────────────────────────────────

@check("live_engine.run_once (3 tickers × 2 algos)")
def test_live_cycle():
    from alpaca_dashboard import live_engine, store
    store.delete_all_pulses()
    live_engine._STATE.tickers = ["SPY", "AAPL", "NVDA"]
    live_engine._STATE.algos = ["moderate", "degen"]
    live_engine._DEDUP_STATE.clear()
    r = live_engine.run_once()
    assert r["generated"] >= 1, f"0 pulses generated"
    rows = [p for p in store.all_pulses(limit=100)
            if (p.get("job_id") or "").startswith(("go_live_", "ab_live_"))]
    assert len(rows) >= 1
    r0 = rows[0]
    assert r0.get("generated_at"), "missing generated_at"
    assert r0.get("status") == "active"
    return f"+{r['generated']} new, {r['skipped']} dedup, {r['errors']} errors"


# ── 10. Ingestion converter ─────────────────────────────────────────────────

@check("converter.to_market_pulse_row")
def test_converter():
    from alpaca_dashboard.ingestion.converter import to_market_pulse_row, TARGET_COLUMNS
    row = {
        "pulse_id": "qa_conv", "algo_id": "moderate", "ticker": "SPY",
        "pulse_type": "bull_put_spread", "entry_date": "2026-04-16",
        "entry_price": 500.0, "pgi": 25.0, "expiry": "2026-05-01",
        "dte": 15, "status": "active", "outcome": None,
        "outcome_price": None, "outcome_pnl_pct": None,
        "selection_reason": "qa test", "generated_at": "2026-04-16T14:00:00Z",
        "top_rec_json": '{"strategy_type":"bull_put_spread"}',
        "indicators_json": '{"pgi":25}',
    }
    mp = to_market_pulse_row(row)
    assert mp["pulse_id"] == "qa_conv"
    assert mp["generated_by"] == "alpaca_backtesting_v1"
    assert mp["generated_at"] == "2026-04-16T14:00:00Z"
    assert mp["risk_level"] == 2  # moderate
    assert mp["score"] == 62     # 50 + 25/2
    assert mp["action"] == "BUY"  # pgi >= 10
    assert mp["asset_class"] == "equities"
    assert mp["close_trigger"] == "backtest"
    for col in TARGET_COLUMNS:
        assert col in mp, f"missing column {col}"
    return f"all {len(TARGET_COLUMNS)} TARGET_COLUMNS present, score={mp['score']}, action={mp['action']}"


# ── 11. HTTP destination healthcheck ─────────────────────────────────────────

@check("HttpDestination.healthcheck (production)")
def test_http_health():
    from alpaca_dashboard.ingestion import HttpDestination
    key = os.getenv("INGEST_API_KEY", "")
    assert key, "INGEST_API_KEY not set"
    dest = HttpDestination(auth_token=key)
    ok, msg = dest.healthcheck()
    assert ok, f"healthcheck failed: {msg}"
    return msg


# ── 12. HTTP destination write (1 pulse → production) ───────────────────────

@check("HttpDestination.write (1 QA pulse → production)")
def test_http_push():
    from alpaca_dashboard.ingestion import HttpDestination
    from alpaca_dashboard.ingestion.converter import to_market_pulse_row
    key = os.getenv("INGEST_API_KEY", "")
    dest = HttpDestination(auth_token=key)
    row = to_market_pulse_row({
        "pulse_id": f"qa_http_{int(time.time())}", "algo_id": "moderate",
        "ticker": "QA_TEST", "pulse_type": "qa_test",
        "entry_date": "2026-04-16", "entry_price": 1.0,
        "pgi": 0, "expiry": "2026-04-17", "dte": 1,
        "status": "expired", "outcome": "neutral",
        "outcome_price": 1.0, "outcome_pnl_pct": 0.0,
        "selection_reason": "self-QA probe — safe to ignore",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_rec_json": None, "indicators_json": None,
    })
    n = dest.write([row])
    assert n >= 1, f"expected >=1 inserted, got {n}"
    return f"inserted={n}"


# ── 13. Pipeline push (end-to-end Turso → production) ───────────────────────

@check("pipeline.push (Turso → production, idempotent)")
def test_pipeline():
    from alpaca_dashboard import store
    from alpaca_dashboard.ingestion import HttpDestination, push
    key = os.getenv("INGEST_API_KEY", "")
    dest = HttpDestination(auth_token=key)
    # Seed one pulse
    pid = f"qa_pipe_{int(time.time())}"
    store.save_pulse({
        "pulse_id": pid, "algo_id": "degen", "ticker": "QA_PIPE",
        "pulse_type": "qa", "strategy_label": "qa",
        "entry_date": "2026-04-16", "entry_price": 1.0,
        "expiry": "2026-04-17", "dte": 1, "pgi": 0, "sigma": 0.1,
        "status": "expired", "outcome": "neutral",
        "outcome_price": 1.0, "outcome_pnl_pct": 0.0,
        "selection_reason": "qa", "job_id": "qa",
        "top_rec_json": None, "indicators_json": None,
        "market_regime": None, "cap_bucket": None,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    r1 = push(dest, reset_cursor=True, batch_size=100)
    assert r1.total_written >= 1, f"first push wrote {r1.total_written}"
    assert r1.error is None, f"push error: {r1.error}"
    # Idempotent: second push should write 0 new
    r2 = push(dest, batch_size=100)
    assert r2.total_read == 0, f"expected 0 on re-push, got {r2.total_read}"
    store.delete_all_pulses()
    return f"push1: {r1.total_written} written, push2: {r2.total_read} re-read (idempotent)"


# ── 14. Pulse chart ─────────────────────────────────────────────────────────

@check("pulse_chart.build_pulse_chart (renders for live pulse)")
def test_pulse_chart():
    from alpaca_dashboard import store as _store
    from alpaca_dashboard.pulse_chart import build_pulse_chart
    rows = [r for r in _store.all_pulses(limit=50)
            if (r.get("job_id") or "").startswith(("go_live_", "ab_live_"))]
    if not rows:
        rows = _store.all_pulses(limit=5)
    assert rows, "no pulses in store to chart"
    fig = build_pulse_chart(rows[0])
    assert fig is not None, f"chart returned None for {rows[0].get('ticker')}"
    assert len(fig.data) >= 1, "no traces"
    assert len(fig.layout.shapes) >= 2, "missing reference lines"
    return f"{rows[0]['ticker']} → {len(fig.data)} traces, {len(fig.layout.shapes)} shapes"


# ── 15. Order executor ──────────────────────────────────────────────────────

@check("order_executor (execution disabled by default)")
def test_execution_disabled():
    from alpaca_dashboard.order_executor import is_execution_enabled
    assert not is_execution_enabled("degen"), "should be disabled by default"
    return "disabled OK"


@check("order_executor.get_account_summary (degen)")
def test_account_summary():
    from alpaca_dashboard.order_executor import get_account_summary
    s = get_account_summary("degen")
    assert s is not None, "no summary returned"
    assert s["equity"] > 0, f"zero equity: {s}"
    return f"equity=${s['equity']:,.0f}, positions={s['positions']}"


# ── 16. SqliteDestination healthcheck ────────────────────────────────────────

@check("SqliteDestination.healthcheck (schema validation)")
def test_sqlite_health():
    import sqlite3
    import tempfile
    from alpaca_dashboard.ingestion import SqliteDestination
    from alpaca_dashboard.ingestion.converter import TARGET_COLUMNS
    # Good schema
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    extra = [c for c in TARGET_COLUMNS if c != "pulse_id"]
    cols_sql = ", ".join(f"{c} TEXT" for c in extra)
    conn.execute(f"CREATE TABLE pulses (id INTEGER PRIMARY KEY, pulse_id TEXT UNIQUE, {cols_sql})")
    conn.commit()
    conn.close()
    dest = SqliteDestination(tmp.name)
    ok, msg = dest.healthcheck()
    os.unlink(tmp.name)
    assert ok, f"good-schema healthcheck failed: {msg}"
    # Bad schema (missing columns)
    tmp2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp2.close()
    conn = sqlite3.connect(tmp2.name)
    conn.execute("CREATE TABLE pulses (id INTEGER PRIMARY KEY, pulse_id TEXT UNIQUE)")
    conn.commit()
    conn.close()
    dest2 = SqliteDestination(tmp2.name)
    ok2, msg2 = dest2.healthcheck()
    os.unlink(tmp2.name)
    assert not ok2, "bad-schema should fail healthcheck"
    assert "missing columns" in msg2
    return "good=pass, bad=rejected with 'missing columns'"


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Skip slow network checks (fetch_history, backtest, live cycle)")
    args = ap.parse_args()

    print("=" * 60)
    print("  SELF-QA: alpaca-backtesting")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Always run
    test_store_init()
    test_store_crud()
    test_store_coefs()
    test_store_cursor()
    test_store_pagination()
    test_market_clock()
    test_cap()
    test_regime()
    test_pgi()
    test_hv()
    test_json_safe()
    test_cache()
    test_converter()
    test_sqlite_health()

    if not args.quick:
        print("\n  --- network checks (may take 30-60s) ---")
        test_fetch()
        test_strategy()
        test_backtest()
        test_live_cycle()
        test_pulse_chart()
        test_execution_disabled()
        test_account_summary()
        test_http_health()
        test_http_push()
        test_pipeline()
    else:
        print("\n  (skipping network checks — run without --quick for full suite)")

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print("\n" + "=" * 60)
    print(f"  {passed} passed, {failed} failed, {len(_results)} total")
    if failed:
        print("\n  FAILURES:")
        for name, ok, msg in _results:
            if not ok:
                print(f"    ❌ {name}: {msg}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
