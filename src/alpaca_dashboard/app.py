"""Streamlit dashboard entrypoint.

Run: ``streamlit run src/alpaca_dashboard/app.py``
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .alpaca_client import AlpacaMultiClient
from .settings import algo_to_account, load_accounts, load_algos

st.set_page_config(page_title="Alpaca Algo Dashboard", layout="wide", page_icon="📊")

ACCOUNTS = load_accounts()
ALGOS = load_algos()
ALGO_ACC = algo_to_account(ACCOUNTS)
CLIENT = AlpacaMultiClient(ACCOUNTS)


# ───────────────────────── sidebar ─────────────────────────
st.sidebar.title("📊 Alpaca Algo Dashboard")
page = st.sidebar.radio("View", ["Overview", "Accounts", "Algos", "Orders"])

with st.sidebar.expander("Configuration status", expanded=False):
    for acc in ACCOUNTS:
        ok = acc.is_configured
        st.write(f"{'🟢' if ok else '🔴'} **{acc.name}** (`{acc.id}`)")
        if not ok:
            st.caption(f"Missing `{acc.env_prefix}_KEY` / `_SECRET` in .env")


# ───────────────────────── pages ─────────────────────────
def page_overview() -> None:
    st.header("Overview")

    snapshots = [CLIENT.snapshot(a.id) for a in ACCOUNTS]
    configured = [s for s in snapshots if s]

    if not configured:
        st.warning("No accounts configured. Add keys to `.env` and reload.")
        return

    cols = st.columns(len(configured))
    for col, snap in zip(cols, configured):
        with col:
            st.metric(
                label=f"{snap.name}",
                value=f"${snap.equity:,.2f}",
                delta=f"{snap.pct_change_today:+.2f}%",
            )
            st.caption(
                f"Cash ${snap.cash:,.0f} · BP ${snap.buying_power:,.0f} · "
                f"{snap.positions_count} pos · {snap.status}"
            )

    st.divider()
    st.subheader("Algo → Account mapping")
    rows = [
        {
            "Algo": f"{a.emoji} {a.name}",
            "Risk": a.risk,
            "DTE": f"{a.dte_min}–{a.dte_max}",
            "Size %": f"{a.position_size_pct*100:.1f}%",
            "Account": ALGO_ACC[a.id].name if a.id in ALGO_ACC else "—",
            "Enabled": "✅" if a.enabled else "⏸",
        }
        for a in ALGOS
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_accounts() -> None:
    st.header("Accounts")

    tabs = st.tabs([a.name for a in ACCOUNTS])
    for tab, acc in zip(tabs, ACCOUNTS):
        with tab:
            st.caption(acc.description)
            snap = CLIENT.snapshot(acc.id)
            if not snap:
                st.info(f"No credentials for {acc.name}. Set `{acc.env_prefix}_KEY/_SECRET` in `.env`.")
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Equity", f"${snap.equity:,.2f}", f"{snap.pct_change_today:+.2f}%")
            c2.metric("Cash", f"${snap.cash:,.2f}")
            c3.metric("Buying power", f"${snap.buying_power:,.2f}")
            c4.metric("Positions", snap.positions_count)

            st.subheader("Positions")
            positions = CLIENT.positions(acc.id)
            if positions:
                df = pd.DataFrame(positions)
                cols = [c for c in [
                    "symbol", "qty", "side", "avg_entry_price", "current_price",
                    "market_value", "unrealized_pl", "unrealized_plpc",
                ] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No open positions.")


def page_algos() -> None:
    st.header("Algos")

    for algo in ALGOS:
        with st.expander(f"{algo.emoji} {algo.name}  ·  {algo.risk}  ·  {algo.dte_min}–{algo.dte_max} DTE", expanded=False):
            acc = ALGO_ACC.get(algo.id)
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe target", f"{algo.sharpe_target:.2f}")
            c2.metric("PoP range", f"{algo.pop_range[0]*100:.0f}–{algo.pop_range[1]*100:.0f}%")
            c3.metric("Position size", f"{algo.position_size_pct*100:.1f}%")

            st.write(f"**Account:** {acc.name if acc else '—'}")
            st.write(f"**PGI gate:** `{algo.pgi_gate}`")
            st.write(f"**Universe:** `{algo.universe}`")
            st.write("**Strategies:** " + ", ".join(f"`{s}`" for s in algo.strategies))

            # Per-algo order count (from live Alpaca, filtered by client_order_id prefix)
            if acc and acc.is_configured:
                orders = CLIENT.orders(acc.id, status="all", limit=500)
                algo_orders = [
                    o for o in orders
                    if CLIENT.algo_from_client_order_id(o.get("client_order_id")) == algo.id
                ]
                st.caption(f"{len(algo_orders)} orders attributed to this algo.")


def page_orders() -> None:
    st.header("Orders")

    acc_id = st.selectbox("Account", [a.id for a in ACCOUNTS], format_func=lambda x: CLIENT.account(x).name)
    status = st.selectbox("Status", ["all", "open", "closed"], index=0)
    limit = st.slider("Limit", 10, 500, 100, step=10)

    orders = CLIENT.orders(acc_id, status=status, limit=limit)
    if not orders:
        st.info("No orders found.")
        return

    df = pd.DataFrame(orders)
    df["algo"] = df.get("client_order_id", pd.Series()).map(CLIENT.algo_from_client_order_id)
    cols = [c for c in [
        "submitted_at", "symbol", "side", "qty", "filled_qty",
        "order_type", "status", "filled_avg_price", "algo", "client_order_id",
    ] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


PAGES = {
    "Overview": page_overview,
    "Accounts": page_accounts,
    "Algos": page_algos,
    "Orders": page_orders,
}
PAGES[page]()
