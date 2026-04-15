"""/Remote_Control — simplified one-tap control surface.

Goal: give the user (or anyone with the app link) a pared-down page to
kick off backtests, stop them, and watch progress — without wading through
the dense /Admin sliders. Optimised for a phone or a second monitor.

Streamlit reruns the page on every interaction, so we rely on the
``jobs`` module's module-global registry (thread handles survive reruns)
and ``store`` for persisted progress.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from alpaca_dashboard import jobs, store
from alpaca_dashboard.backtest import DEFAULT_TICKERS
from alpaca_dashboard.settings import load_algos

st.set_page_config(page_title="Remote Control", layout="wide", page_icon="🎮")
store.init_db()

ALGOS = load_algos()
READY = [a for a in ALGOS if a.is_ready]
PLANNED = [a for a in ALGOS if not a.is_ready]

# ── Header + global controls ─────────────────────────────────────────────────
st.title("🎮 Remote Control")
st.caption(
    "Tap an algo to fire one pulse batch. Batches run in the background — "
    "you can close the tab; they keep going. Live status below."
)

live_jobs = {j["algo_id"]: j for j in jobs.snapshot() if j["alive"]}

# Global defaults. Power users want /Admin; this page hides complexity.
DEFAULT_BATCH = 50
DEFAULT_DAYS = 365
TICKER_SHORTLIST = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "META", "TSLA", "AMZN"]

c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    size = st.radio(
        "Batch size",
        options=[20, 50, 100, 200],
        index=1,
        horizontal=True,
        help="How many pulses to generate per tap. Larger = longer wait.",
    )
with c_top2:
    if st.button("⏹ Stop all running", width="stretch",
                 disabled=not live_jobs):
        for job in live_jobs.values():
            jobs.request_stop(job["job_id"])
        st.toast(f"Stop requested for {len(live_jobs)} job(s)")
        time.sleep(0.3)
        st.rerun()

st.divider()

# ── Big per-algo cards ───────────────────────────────────────────────────────
st.subheader("Ready algos")

# 3-column grid; each cell is one algo card
rows = [READY[i:i + 3] for i in range(0, len(READY), 3)]
for row in rows:
    cols = st.columns(len(row))
    for col, algo in zip(cols, row):
        with col.container(border=True):
            st.markdown(f"### {algo.emoji} {algo.name}")
            st.caption(
                f"{algo.risk} · DTE {algo.dte_min}–{algo.dte_max} · "
                f"Sharpe target {algo.sharpe_target:.2f}"
            )

            live = live_jobs.get(algo.id)
            stored = len(store.pulses_for_algo(algo.id, limit=100_000))

            if live:
                p = live["progress"]
                pct = (p["done"] / p["target"]) if p.get("target") else 0
                st.progress(
                    min(pct, 1.0),
                    text=f"🏃 {p['done']}/{p['target']} · {p.get('msg', '')[:30]}",
                )
                if st.button("⏹ Stop", key=f"stop_{algo.id}",
                             width="stretch"):
                    jobs.request_stop(live["job_id"])
                    st.rerun()
            else:
                # Last completed summary, if any
                recent = [j for j in jobs.snapshot()
                          if j["algo_id"] == algo.id and j.get("summary")][:1]
                if recent and recent[0].get("summary"):
                    s = recent[0]["summary"]
                    pf = s.get("profit_factor")
                    st.caption(
                        f"Last run · {s.get('total', 0)} trades · "
                        f"WR {s.get('win_rate', 0):.0f}% · "
                        f"PF {pf:.2f}" if pf else
                        f"Last run · {s.get('total', 0)} trades · "
                        f"WR {s.get('win_rate', 0):.0f}%"
                    )
                else:
                    st.caption("Never run.")

                if st.button(f"▶ Run {size}",
                             key=f"run_{algo.id}",
                             type="primary",
                             width="stretch"):
                    jobs.start_algo_job(
                        algo_id=algo.id,
                        tickers=TICKER_SHORTLIST,
                        days=DEFAULT_DAYS,
                        target_pulses=int(size),
                    )
                    st.toast(f"▶ {algo.name} — {size} pulses queued")
                    time.sleep(0.3)
                    st.rerun()

            st.caption(f"💾 {stored:,} total pulses stored")

# ── Planned algos — read-only strip ──────────────────────────────────────────
if PLANNED:
    st.divider()
    st.subheader("Planned algos")
    st.caption("Strategy selectors TBD. Run is disabled until status flips to ready.")
    planned_cols = st.columns(len(PLANNED))
    for col, algo in zip(planned_cols, PLANNED):
        with col.container(border=True):
            st.markdown(f"### {algo.emoji} {algo.name}")
            st.caption(f"{algo.risk} · DTE {algo.dte_min}–{algo.dte_max}")
            st.caption(f"Universe · `{algo.universe}`")
            st.button("▶ Run (disabled)", key=f"rc_planned_{algo.id}",
                      disabled=True, width="stretch")

# ── Auto-refresh while any run is live ───────────────────────────────────────
if live_jobs:
    time.sleep(2)
    st.rerun()
