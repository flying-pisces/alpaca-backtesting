"""/Admin — run, stop, and tune backtests per algo.

Left column: per-algo job controls (run N pulses, stop, recent status).
Right column: coefficient sliders (pgi_entry, target_dte, size_mult).

Notes on statefulness: Streamlit reruns the page on every interaction, but
the ``jobs`` module holds thread handles in a module-global registry, so
background runs survive across reruns until they finish.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from alpaca_dashboard import jobs, live_engine, store
from alpaca_dashboard.backtest import ALGOS as READY_ALGO_IDS, DEFAULT_TICKERS
from alpaca_dashboard.settings import load_accounts, load_algos

st.set_page_config(page_title="Admin · Run algos", layout="wide", page_icon="⚙️")
store.init_db()

ALGOS = load_algos()
ALGO_BY_ID = {a.id: a for a in ALGOS}
ACCOUNTS = load_accounts()

st.title("⚙️ Admin — backtest runner")
st.caption(
    "Trigger walk-forward backtests per algo, push multiple pulses into storage, "
    "stop a run mid-flight, and tune per-algo coefficients used on subsequent runs."
)

# ── Global run settings ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Run settings")
    days = st.slider("History window (days)", 90, 730, 365, step=30)
    target_pulses = st.slider("Target pulses per run", 20, 500, 120, step=10)
    tickers_input = st.text_area(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
        height=140,
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.divider()
    st.caption("Danger zone")
    if st.button("Delete ALL pulses", type="secondary"):
        n = store.delete_all_pulses()
        st.warning(f"Deleted {n} rows.")

# ── Per-algo controls ────────────────────────────────────────────────────────
live_jobs = {j["algo_id"]: j for j in jobs.snapshot() if j["alive"]}

ready_algos = [a for a in ALGOS if a.is_ready]
planned_algos = [a for a in ALGOS if not a.is_ready]

st.header(f"Ready algos · {len(ready_algos)}")

for algo in ready_algos:
    with st.expander(
        f"{algo.emoji} **{algo.name}**  ·  {algo.risk}  ·  DTE {algo.dte_min}–{algo.dte_max}",
        expanded=(algo.id in live_jobs),
    ):
        cols = st.columns([2, 2, 3])

        # ── Column 1: job control ───────────────────────────────────────────
        with cols[0]:
            st.subheader("Run")
            live = live_jobs.get(algo.id)
            if live:
                p = live["progress"]
                pct = (p["done"] / p["target"]) if p.get("target") else 0
                st.progress(min(pct, 1.0),
                            text=f"{p['done']}/{p['target']} — {p.get('msg','')}")
                if st.button("Stop", key=f"stop_{algo.id}"):
                    jobs.request_stop(live["job_id"])
                    st.rerun()
            else:
                n_runs = st.number_input(
                    "How many pulse batches?",
                    min_value=1, max_value=10, value=1, step=1,
                    key=f"batch_{algo.id}",
                    help="Each batch runs the configured target_pulses back-to-back.",
                )
                if st.button("▶ Run", key=f"run_{algo.id}", type="primary"):
                    for i in range(int(n_runs)):
                        jobs.start_algo_job(
                            algo_id=algo.id,
                            tickers=tickers,
                            days=days,
                            target_pulses=target_pulses,
                        )
                    st.success(f"Queued {n_runs} run(s) for {algo.name}")
                    time.sleep(0.5)
                    st.rerun()

            total_stored = len([p for p in store.pulses_for_algo(algo.id, limit=50000)])
            st.caption(f"{total_stored:,} pulses stored for {algo.name}")
            if st.button("🗑 Clear pulses for this algo", key=f"clr_{algo.id}"):
                store.delete_pulses_for_algo(algo.id)
                st.rerun()

        # ── Column 2: coefficients ──────────────────────────────────────────
        with cols[1]:
            st.subheader("Coefficients")
            current = store.get_coefficients(algo.id)
            default_dte = int((algo.dte_min + algo.dte_max) / 2)
            target_dte = st.slider(
                "target_dte (days to expiry to aim for)",
                max(1, algo.dte_min), max(algo.dte_max, algo.dte_min + 1),
                int(current.get("target_dte", default_dte)),
                key=f"dte_{algo.id}",
            )
            pgi_entry = st.slider(
                "pgi_entry (min |PGI| to enter)",
                0.0, 80.0,
                float(current.get("pgi_entry", 0.0)),
                step=5.0,
                key=f"pgi_{algo.id}",
            )
            size_mult = st.slider(
                "size_mult (scales position sizing)",
                0.25, 3.0,
                float(current.get("size_mult", 1.0)),
                step=0.25,
                key=f"sm_{algo.id}",
            )
            if st.button("Save", key=f"save_{algo.id}"):
                store.set_coefficients(algo.id, {
                    "target_dte": target_dte,
                    "pgi_entry": pgi_entry,
                    "size_mult": size_mult,
                })
                st.success("Saved — applies to subsequent runs.")
                st.rerun()

        # ── Column 3: last-run summary ──────────────────────────────────────
        with cols[2]:
            st.subheader("Recent runs")
            algo_jobs = [j for j in jobs.snapshot() if j["algo_id"] == algo.id][:5]
            if not algo_jobs:
                st.caption("No runs yet.")
            for j in algo_jobs:
                status = j["status"]
                icon = {"running": "🏃", "done": "✅", "stopped": "⏹",
                        "error": "❌"}.get(status, "•")
                started = j["started_at"][:19].replace("T", " ") if j["started_at"] else ""
                st.markdown(f"{icon} `{j['job_id']}` — **{status}** · {started}")
                if j.get("summary"):
                    s = j["summary"]
                    st.caption(
                        f"{s.get('total',0)} trades · WR {s.get('win_rate',0):.1f}% · "
                        f"avg {s.get('avg_pnl',0):+.2f}% · "
                        f"PF {s.get('profit_factor') or '—'}"
                    )
                if j.get("error"):
                    st.error(j["error"])

# ── Planned algos — placeholder-only, runs blocked ───────────────────────────
if planned_algos:
    st.divider()
    st.header(f"Planned algos · {len(planned_algos)}")
    st.caption(
        "Accounts are live (you can see equity on Home), but the strategy "
        "selector has not been implemented. Once wired into "
        "`select_strategy_for_tier`, flip `status: ready` in "
        "`config/algos.yaml` to enable the Run button."
    )
    for algo in planned_algos:
        with st.expander(
            f"{algo.emoji} **{algo.name}**  ·  PLANNED  ·  DTE {algo.dte_min}–{algo.dte_max}",
            expanded=False,
        ):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Intent**")
                st.caption(f"Risk · {algo.risk}")
                st.caption(f"Universe · `{algo.universe}`")
                st.caption(f"PGI gate · `{algo.pgi_gate}`")
                st.caption(f"Sharpe target · {algo.sharpe_target:.2f}")
            with c2:
                st.markdown("**Checklist to promote to `ready`**")
                st.markdown(
                    "- [ ] Implement signal scanner for "
                    f"`{algo.universe}`\n"
                    "- [ ] Add risk_tier branch to "
                    f"`select_strategy_for_tier(..., risk_tier='{algo.id}')`\n"
                    "- [ ] Define strategy list (currently empty)\n"
                    "- [ ] Flip `status: ready` + `enabled: true` in "
                    "`config/algos.yaml`\n"
                    "- [ ] Run a 20-pulse dry backtest to sanity-check P&L"
                )
            st.button("▶ Run (disabled)", key=f"run_planned_{algo.id}",
                      disabled=True, help="Algo is planned — flip status to ready first.")

# ── Footer: all-jobs table ───────────────────────────────────────────────────
st.divider()
st.subheader("All jobs")
all_jobs = jobs.snapshot()
if all_jobs:
    jt = pd.DataFrame([
        {
            "job_id": j["job_id"],
            "algo": j["algo_id"],
            "status": j["status"],
            "started": j["started_at"],
            "ended": j["ended_at"],
            "done": j["progress"].get("done", 0),
            "target": j["progress"].get("target", 0),
            "summary": j["summary"],
        }
        for j in all_jobs
    ])
    st.dataframe(jt, width="stretch", hide_index=True)
else:
    st.caption("No jobs in this session yet.")

# ── Live engine ──────────────────────────────────────────────────────────────
st.divider()
st.header("🔴 Live engine — Go Live Pulses")
st.caption(
    "Scans the universe every N seconds and emits pulses tagged "
    "`go_live_<algo>`. Shares the same schema as backtest pulses; "
    "ingestion converter handles both. Feed is on **/Live**."
)

engine_state = live_engine.state_snapshot()

le_c1, le_c2, le_c3 = st.columns([3, 2, 2])
with le_c1:
    live_tickers_text = st.text_area(
        "Tickers",
        value=",".join(engine_state["tickers"]),
        height=80,
        key="live_tickers",
    )
    live_tickers = [t.strip().upper() for t in live_tickers_text.split(",") if t.strip()]
    live_algos = st.multiselect(
        "Ready algos to scan",
        options=READY_ALGO_IDS,
        default=engine_state["algos"] or READY_ALGO_IDS,
        key="live_algos",
    )
with le_c2:
    cycle_sec = st.slider(
        "Cycle (seconds between full scans)",
        min_value=30, max_value=1800,
        value=int(engine_state["cycle_sec"]),
        step=30,
    )
    dedup_sec = st.slider(
        "Dedup window (seconds)",
        min_value=60, max_value=7200,
        value=int(engine_state["dedup_sec"]),
        step=60,
    )
with le_c3:
    dot = "🟢 running" if engine_state["running"] else "⚪️ stopped"
    st.markdown(f"### {dot}")
    st.caption(
        f"last cycle: {(engine_state['last_cycle_at'] or '—')[:19].replace('T',' ')} · "
        f"+{engine_state['last_cycle_generated']} new / "
        f"{engine_state['last_cycle_skipped_dedup']} dedup"
    )
    st.caption(
        f"total: {engine_state['total_generated']} pulses · "
        f"{engine_state['total_cycles']} cycles · "
        f"cache hit {engine_state['cache'].get('hit_rate', 0)*100:.0f}%"
    )

btn_c1, btn_c2, btn_c3 = st.columns(3)
with btn_c1:
    if st.button(
        "▶ Start engine",
        type="primary",
        disabled=engine_state["running"] or not live_tickers or not live_algos,
        width="stretch",
    ):
        live_engine.start(
            tickers=live_tickers,
            algos=live_algos,
            cycle_sec=cycle_sec,
            dedup_sec=dedup_sec,
        )
        st.toast(f"🔴 Live engine started — {len(live_algos)} algos × {len(live_tickers)} tickers")
        time.sleep(0.3)
        st.rerun()
with btn_c2:
    if st.button(
        "⏹ Stop engine",
        disabled=not engine_state["running"],
        width="stretch",
    ):
        live_engine.stop()
        st.toast("engine stopping…")
        time.sleep(0.3)
        st.rerun()
with btn_c3:
    if st.button("🔄 Run one cycle now", width="stretch"):
        with st.spinner("scanning universe…"):
            # Apply current UI values even if not fully started
            live_engine._STATE.tickers = live_tickers or live_engine._STATE.tickers
            live_engine._STATE.algos = live_algos or live_engine._STATE.algos
            r = live_engine.run_once()
        st.success(
            f"+{r['generated']} new · {r['skipped']} deduped · "
            f"{r['errors']} errors"
        )
        time.sleep(0.4)
        st.rerun()

if engine_state.get("last_error"):
    st.error(f"last engine error: {engine_state['last_error']}")


# ── Push to market_pulse ─────────────────────────────────────────────────────
from alpaca_dashboard.ingestion import HttpDestination, SqliteDestination, push as push_to_market_pulse

st.divider()
st.header("📤 Push pulses to market_pulse")
st.caption(
    "Stream new pulses from Turso into market_pulse. "
    "HTTP → production Fly.io (preferred, no lock contention). "
    "SQLite → local dev. Idempotent; cursor-tracked."
)

push_c1, push_c2 = st.columns([3, 2])
with push_c1:
    dest_mode = st.radio(
        "Destination",
        ["HTTP (production)", "SQLite (local)"],
        horizontal=True,
        index=0 if os.getenv("INGEST_API_KEY") else 1,
    )
    if dest_mode.startswith("HTTP"):
        ingest_key = os.getenv("INGEST_API_KEY", "")
        st.text_input(
            "INGEST_API_KEY",
            value="•" * min(len(ingest_key), 20) if ingest_key else "",
            disabled=True,
            help="Set via .env or Streamlit Cloud Secrets.",
        )
        dest_obj = HttpDestination(auth_token=ingest_key)
    else:
        dest_path = st.text_input(
            "market_pulse pulses.db path",
            value="~/projects/market_pulse/pulses.db",
        )
        dest_obj = SqliteDestination(dest_path)

    push_algo = st.selectbox(
        "Restrict to algo (optional)",
        options=["(all)"] + [a.id for a in ALGOS if a.is_ready],
        index=0,
    )
    push_reset = st.checkbox(
        "Reset cursor (ignore last-pushed position — use for initial backfill)",
        value=False,
    )

with push_c2:
    try:
        ok, msg = dest_obj.healthcheck()
        if ok:
            st.success(f"✅ {msg[:80]}")
        else:
            st.error(f"Destination: {msg}")
    except Exception as e:
        st.error(f"Bad destination: {e}")
        ok = False

    existing_cursor = None
    try:
        existing_cursor = store.get_ingestion_cursor(dest_obj.name)
    except Exception:  # noqa: BLE001
        pass
    if existing_cursor:
        st.caption(
            f"Last push: **{existing_cursor.get('last_pushed_count', 0)}** rows · "
            f"cursor `{(existing_cursor.get('last_pushed_created_at') or '')[:19]}`"
        )
    else:
        st.caption("No prior push to this destination.")

    run_push = st.button("▶ Push now", type="primary", disabled=not ok)

if run_push:
    with st.spinner("pushing…"):
        result = push_to_market_pulse(
            dest_obj,
            algo_id=None if push_algo == "(all)" else push_algo,
            batch_size=200,
            reset_cursor=push_reset,
        )
    if result.error:
        st.error(f"Push failed: {result.error}")
    else:
        st.success(
            f"Pushed {result.total_written}/{result.total_read} rows in "
            f"{result.batches} batches · {result.elapsed_sec:.1f}s"
        )
    st.json(result.as_dict())

# Auto-refresh while anything is running
if any(j["alive"] for j in all_jobs):
    time.sleep(2)
    st.rerun()
