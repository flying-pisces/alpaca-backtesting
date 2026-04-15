"""Home-page renderer. Kept as a function so Streamlit reruns work on Cloud.

Streamlit Cloud reruns ``streamlit_app.py`` on every interaction. If the
page content lived at module top-level here it would only render once per
process (Python caches modules in sys.modules, so ``from alpaca_dashboard
import app`` on subsequent reruns is a no-op). Calling ``render()`` from
``streamlit_app.py`` makes the script explicitly re-execute the widgets.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from alpaca_dashboard.alpaca_client import AlpacaMultiClient
from alpaca_dashboard.settings import algo_to_account, load_accounts, load_algos
from alpaca_dashboard import store


def render() -> None:
    st.set_page_config(page_title="Alpaca Algo Dashboard", layout="wide", page_icon="📊")
    store.init_db()

    accounts = load_accounts()
    algos = load_algos()
    algo_acc = algo_to_account(accounts)
    client = AlpacaMultiClient(accounts)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📊 Alpaca Algo Dashboard")
    st.sidebar.caption("Home · **Dashboard** · **Admin**  (see page list above)")

    with st.sidebar.expander("Configuration status", expanded=False):
        for acc in accounts:
            ok = acc.is_configured
            st.write(f"{'🟢' if ok else '🔴'} **{acc.name}** (`{acc.id}`)")
            if not ok:
                st.caption(f"Missing `{acc.env_prefix}_KEY` / `_SECRET` in .env")

    # ── Algo tables (no network I/O — render first) ──────────────────────────
    ready_algos = [a for a in algos if a.is_ready]
    planned_algos = [a for a in algos if not a.is_ready]

    def _algo_row(a) -> dict:
        return {
            "Algo": f"{a.emoji} {a.name}",
            "Risk": a.risk,
            "DTE": f"{a.dte_min}–{a.dte_max}",
            "Size %": f"{a.position_size_pct*100:.1f}%",
            "Sharpe target": f"{a.sharpe_target:.2f}",
            "Account": algo_acc[a.id].name if a.id in algo_acc else "—",
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

    # ── Live Alpaca snapshots — parallel + per-call timeout ──────────────────
    st.header("Live paper accounts")

    configured = [a for a in accounts if a.is_configured]
    if not configured:
        st.warning("No accounts configured. Add keys to `.env` and reload.")
        return

    @st.cache_data(ttl=60, show_spinner=False)
    def _snapshot(acc_id: str):
        return client.snapshot(acc_id)

    snaps: dict = {}
    errors: dict = {}
    with st.spinner("Fetching live balances…"):
        with ThreadPoolExecutor(max_workers=len(configured)) as ex:
            futures = {ex.submit(_snapshot, a.id): a for a in configured}
            for fut in as_completed(futures, timeout=15):
                a = futures[fut]
                try:
                    snaps[a.id] = fut.result(timeout=5)
                except Exception as e:   # noqa: BLE001
                    errors[a.id] = f"{type(e).__name__}: {e}"

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


# Allow ``streamlit run src/alpaca_dashboard/app.py`` to still work locally.
if __name__ == "__main__" or True:
    # The ``or True`` ensures the body runs both when imported via
    # ``streamlit run src/alpaca_dashboard/app.py`` (where ``__name__`` is
    # ``__main__``) and when this file is opened directly by Streamlit's page
    # manager (where ``__name__`` is the module path). Guarded behind a
    # function so every Streamlit rerun re-invokes render().
    pass
