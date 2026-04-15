"""Streamlit entrypoint — home / overview page.

Run: ``streamlit run src/alpaca_dashboard/app.py``

Routes:
    /               this file (live account overview)
    /Dashboard      backtest results (pages/1_Dashboard.py)
    /Admin          run jobs, tune coefficients (pages/2_Admin.py)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alpaca_dashboard.alpaca_client import AlpacaMultiClient
from alpaca_dashboard.settings import algo_to_account, load_accounts, load_algos
from alpaca_dashboard import store

st.set_page_config(page_title="Alpaca Algo Dashboard", layout="wide", page_icon="📊")
store.init_db()

ACCOUNTS = load_accounts()
ALGOS = load_algos()
ALGO_ACC = algo_to_account(ACCOUNTS)
CLIENT = AlpacaMultiClient(ACCOUNTS)

st.sidebar.title("📊 Alpaca Algo Dashboard")
st.sidebar.caption("Home · **Dashboard** · **Admin**  (see page list above)")

with st.sidebar.expander("Configuration status", expanded=False):
    for acc in ACCOUNTS:
        ok = acc.is_configured
        st.write(f"{'🟢' if ok else '🔴'} **{acc.name}** (`{acc.id}`)")
        if not ok:
            st.caption(f"Missing `{acc.env_prefix}_KEY` / `_SECRET` in .env")

# ── overview ──────────────────────────────────────────────────────────────────
st.header("Live paper accounts")

snapshots = []
for a in ACCOUNTS:
    try:
        snapshots.append(CLIENT.snapshot(a.id))
    except Exception as e:
        st.error(f"{a.name}: {e}")
        snapshots.append(None)

configured = [s for s in snapshots if s]
if not configured:
    st.warning("No accounts configured. Add keys to `.env` and reload.")
else:
    cols = st.columns(len(configured))
    for col, snap in zip(cols, configured):
        with col:
            st.metric(
                label=snap.name,
                value=f"${snap.equity:,.2f}",
                delta=f"{snap.pct_change_today:+.2f}%",
            )
            st.caption(
                f"Cash ${snap.cash:,.0f} · BP ${snap.buying_power:,.0f} · "
                f"{snap.positions_count} pos · {snap.status}"
            )

st.divider()

ready_algos = [a for a in ALGOS if a.is_ready]
planned_algos = [a for a in ALGOS if not a.is_ready]


def _algo_row(a) -> dict:
    return {
        "Algo": f"{a.emoji} {a.name}",
        "Risk": a.risk,
        "DTE": f"{a.dte_min}–{a.dte_max}",
        "Size %": f"{a.position_size_pct*100:.1f}%",
        "Sharpe target": f"{a.sharpe_target:.2f}",
        "Account": ALGO_ACC[a.id].name if a.id in ALGO_ACC else "—",
        "Status": "✅ ready" if a.is_ready else "🟡 planned",
    }


st.subheader(f"Ready algos · {len(ready_algos)}")
st.dataframe(pd.DataFrame([_algo_row(a) for a in ready_algos]),
             width="stretch", hide_index=True)

if planned_algos:
    st.subheader(f"Planned algos · {len(planned_algos)}")
    st.caption("Accounts are live, strategy selectors TBD. Admin run-buttons are disabled.")
    st.dataframe(pd.DataFrame([_algo_row(a) for a in planned_algos]),
                 width="stretch", hide_index=True)

st.divider()
c1, c2 = st.columns(2)
c1.info("📈 **Dashboard** — per-algo backtest results, equity curves, trade log.")
c2.info("⚙️ **Admin** — run/stop algo backtests, push pulses, tune coefficients.")
