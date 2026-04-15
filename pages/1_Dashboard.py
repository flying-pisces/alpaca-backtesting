"""/Dashboard — backtest results visualisation.

Pulls from local SQLite (written by the backtest engine on the /Admin page)
and renders equity curves, per-algo summary cards, strategy breakdown, and
a filterable trade log.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alpaca_dashboard import store
from alpaca_dashboard.settings import load_algos

st.set_page_config(page_title="Dashboard · Backtest", layout="wide", page_icon="📈")
store.init_db()

ALGOS = load_algos()
ALGO_BY_ID = {a.id: a for a in ALGOS}


# ── Load data ─────────────────────────────────────────────────────────────────
rows = store.all_pulses(limit=50000)
if not rows:
    st.title("📈 Backtest Dashboard")
    st.warning("No backtest pulses stored yet. Head to **/Admin** and run an algo.")
    st.stop()

df = pd.DataFrame(rows)
df["entry_date"] = pd.to_datetime(df["entry_date"])
df["outcome_pnl_pct"] = pd.to_numeric(df["outcome_pnl_pct"], errors="coerce").fillna(0.0)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
algos_available = sorted(df["algo_id"].unique())
pick_algos = st.sidebar.multiselect("Algos", algos_available, default=algos_available)
min_date = df["entry_date"].min().date()
max_date = df["entry_date"].max().date()
date_range = st.sidebar.date_input("Entry date range", (min_date, max_date),
                                   min_value=min_date, max_value=max_date)
pick_tickers = st.sidebar.multiselect("Tickers",
                                      sorted(df["ticker"].unique()),
                                      default=[])

mask = df["algo_id"].isin(pick_algos)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
    mask &= df["entry_date"].dt.date >= d0
    mask &= df["entry_date"].dt.date <= d1
if pick_tickers:
    mask &= df["ticker"].isin(pick_tickers)
view = df[mask].copy()

st.title("📈 Backtest Dashboard")
st.caption(f"{len(view):,} trades · {view['ticker'].nunique()} tickers · "
           f"{view['entry_date'].min().date()} → {view['entry_date'].max().date()}")

if view.empty:
    st.info("No trades match these filters.")
    st.stop()

# ── Summary cards ────────────────────────────────────────────────────────────
st.subheader("Per-algo summary")


def _summary(g: pd.DataFrame) -> dict:
    wins = (g["outcome"] == "win").sum()
    losses = (g["outcome"] == "loss").sum()
    total = len(g)
    wr = wins / total * 100 if total else 0
    win_pnl = g.loc[g["outcome"] == "win", "outcome_pnl_pct"].sum()
    loss_pnl = g.loc[g["outcome"] == "loss", "outcome_pnl_pct"].sum()
    pf = abs(win_pnl) / abs(loss_pnl) if loss_pnl else None
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "avg_pnl": g["outcome_pnl_pct"].mean(),
        "total_pnl": g["outcome_pnl_pct"].sum(),
        "profit_factor": pf,
    }


summary_rows = []
cols = st.columns(max(1, min(len(pick_algos), 5)))
for i, algo_id in enumerate(pick_algos):
    g = view[view["algo_id"] == algo_id]
    if g.empty:
        continue
    s = _summary(g)
    a = ALGO_BY_ID.get(algo_id)
    with cols[i % len(cols)]:
        label = f"{a.emoji if a else ''} {a.name if a else algo_id}"
        st.metric(label, f"{s['win_rate']:.1f}% WR",
                  delta=f"{s['total_pnl']:+.1f}% total")
        pf_str = f"{s['profit_factor']:.2f}x" if s["profit_factor"] else "—"
        st.caption(f"{s['total']} trades · avg {s['avg_pnl']:+.2f}% · PF {pf_str}")
    summary_rows.append({"algo_id": algo_id, **s})

# ── Equity curves (cumulative pnl%) ──────────────────────────────────────────
st.subheader("Cumulative P&L (%) by algo")
view_sorted = view.sort_values("entry_date")
view_sorted["cum_pnl"] = view_sorted.groupby("algo_id")["outcome_pnl_pct"].cumsum()

fig = px.line(view_sorted, x="entry_date", y="cum_pnl", color="algo_id",
              labels={"cum_pnl": "Cumulative P&L (%)", "entry_date": "Entry"},
              height=420)
fig.update_layout(legend_title_text="Algo", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ── Drawdown per algo ────────────────────────────────────────────────────────
st.subheader("Drawdown (%)")
drawdown_frames = []
for algo_id, g in view_sorted.groupby("algo_id"):
    cum = g["outcome_pnl_pct"].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    drawdown_frames.append(pd.DataFrame({
        "entry_date": g["entry_date"].values,
        "drawdown": dd.values,
        "algo_id": algo_id,
    }))
if drawdown_frames:
    dd_df = pd.concat(drawdown_frames)
    fig_dd = px.area(dd_df, x="entry_date", y="drawdown", color="algo_id",
                     height=320)
    fig_dd.update_layout(hovermode="x unified",
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Strategy breakdown ───────────────────────────────────────────────────────
st.subheader("Strategy breakdown")
strat_stats = (
    view.groupby(["algo_id", "pulse_type"])
    .agg(trades=("pulse_id", "count"),
         win_rate=("outcome", lambda s: (s == "win").mean() * 100),
         avg_pnl=("outcome_pnl_pct", "mean"))
    .reset_index()
    .sort_values(["algo_id", "trades"], ascending=[True, False])
)
st.dataframe(
    strat_stats.round({"win_rate": 1, "avg_pnl": 2}),
    use_container_width=True, hide_index=True,
)

# ── Win rate heatmap (algo × ticker) ─────────────────────────────────────────
if view["ticker"].nunique() <= 40:
    st.subheader("Win rate by algo × ticker")
    pivot = (
        view.assign(win=(view["outcome"] == "win").astype(int))
            .groupby(["algo_id", "ticker"])["win"]
            .mean()
            .mul(100)
            .unstack("ticker")
    )
    if not pivot.empty:
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn", zmin=0, zmax=100,
            colorbar=dict(title="WR %"),
        ))
        fig_hm.update_layout(height=max(240, 40 * len(pivot.index)))
        st.plotly_chart(fig_hm, use_container_width=True)

# ── Trade log ────────────────────────────────────────────────────────────────
st.subheader("Trade log")
show = view[[
    "entry_date", "algo_id", "ticker", "pulse_type", "strategy_label",
    "dte", "pgi", "entry_price", "outcome", "outcome_price",
    "outcome_pnl_pct", "job_id",
]].sort_values("entry_date", ascending=False)
st.dataframe(show, use_container_width=True, hide_index=True, height=420)

csv = show.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="backtest_trades.csv", mime="text/csv")
