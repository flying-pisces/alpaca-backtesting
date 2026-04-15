"""Streamlit Cloud entrypoint — home page inlined.

Streamlit reruns this file on every interaction. Earlier revisions had the
page code in ``alpaca_dashboard.app`` and imported it here, but:
  1. Python caches modules in ``sys.modules``, so subsequent reruns did
     not re-execute the widget calls → blank page after first render.
  2. If the module crashed during init, Streamlit Cloud cached a partial
     module whose ``render`` attr never got defined → ``ImportError: cannot
     import name 'render'`` that survived across pushes.

The fix is to keep the page body here. Support modules (settings, store,
alpaca_client, strategies, backtest, jobs) stay under ``src/alpaca_dashboard``.
"""
from __future__ import annotations

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


try:
    import pandas as pd
    from alpaca_dashboard.alpaca_client import AlpacaMultiClient
    from alpaca_dashboard.settings import algo_to_account, load_accounts, load_algos
    from alpaca_dashboard import store
except Exception:   # noqa: BLE001
    st.set_page_config(page_title="Error", layout="wide")
    st.error("⚠️ Dependency import failed — full traceback below:")
    st.code(traceback.format_exc(), language="python")
    st.stop()


st.set_page_config(page_title="Alpaca Algo Dashboard", layout="wide", page_icon="📊")
store.init_db()

ACCOUNTS = load_accounts()
ALGOS = load_algos()
ALGO_ACC = algo_to_account(ACCOUNTS)
CLIENT = AlpacaMultiClient(ACCOUNTS)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Alpaca Algo Dashboard")
st.sidebar.caption("Home · **Dashboard** · **Admin**  (see page list above)")

with st.sidebar.expander("Configuration status", expanded=False):
    for acc in ACCOUNTS:
        ok = acc.is_configured
        st.write(f"{'🟢' if ok else '🔴'} **{acc.name}** (`{acc.id}`)")
        if not ok:
            st.caption(f"Missing `{acc.env_prefix}_KEY` / `_SECRET` in .env")

# ── Algo tables (no network I/O — always render first) ──────────────────────
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
    st.caption("Accounts are live, strategy selectors TBD. "
               "Admin run-buttons are disabled.")
    st.dataframe(pd.DataFrame([_algo_row(a) for a in planned_algos]),
                 width="stretch", hide_index=True)

st.divider()
c1, c2 = st.columns(2)
c1.info("📈 **Dashboard** — per-algo backtest results, equity curves, trade log.")
c2.info("⚙️ **Admin** — run/stop algo backtests, push pulses, tune coefficients.")
st.divider()

# ── Live Alpaca snapshots ────────────────────────────────────────────────────
st.header("Live paper accounts")

configured = [a for a in ACCOUNTS if a.is_configured]
if not configured:
    st.warning("No accounts configured. Add keys to `.env` and reload.")
else:
    @st.cache_data(ttl=60, show_spinner=False)
    def _snapshot(acc_id: str):
        return CLIENT.snapshot(acc_id)

    snaps: dict = {}
    errors: dict = {}
    with st.spinner("Fetching live balances…"):
        with ThreadPoolExecutor(max_workers=len(configured)) as ex:
            futures = {ex.submit(_snapshot, a.id): a for a in configured}
            for fut in as_completed(futures, timeout=15):
                acc = futures[fut]
                try:
                    snaps[acc.id] = fut.result(timeout=5)
                except Exception as e:   # noqa: BLE001
                    errors[acc.id] = f"{type(e).__name__}: {e}"

    cols = st.columns(max(1, len(configured)))
    for col, acc in zip(cols, configured):
        with col:
            snap = snaps.get(acc.id)
            if snap is None:
                err = errors.get(acc.id, "no response")
                st.metric(label=acc.name, value="—")
                st.caption(f"⚠️ {err[:60]}")
                continue
            st.metric(
                label=snap.name,
                value=f"${snap.equity:,.2f}",
                delta=f"{snap.pct_change_today:+.2f}%",
            )
            st.caption(
                f"Cash ${snap.cash:,.0f} · BP ${snap.buying_power:,.0f} · "
                f"{snap.positions_count} pos · {snap.status}"
            )
