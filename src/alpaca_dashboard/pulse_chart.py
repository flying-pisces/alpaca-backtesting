"""Interactive pulse visualization matching market_pulse's web/iOS/Android.

Renders a Plotly figure for a single pulse showing:
  1. Historical price line (90-day lookback from entry_date)
  2. Forward price projection to expiry (if still active, uses latest bars)
  3. Entry / target / stop horizontal reference lines
  4. Expected-move cone (σ√t upper and lower bands) from entry to expiry
  5. P&L % annotation badge
  6. Outcome marker if resolved (green ✓ or red ✗)

Palette matches market_pulse:
  bg=#08081A, card=#12122A, win=#2ECC71, loss=#E74C3C, entry=#A0A0CC
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import plotly.graph_objects as go

from .data_cache import fetch_bars

# ── Palette (from market_pulse chart_renderer.py) ────────────────────────────
BG = "#08081A"
CARD = "#12122A"
WIN = "#2ECC71"
LOSS = "#E74C3C"
NEUTRAL = "#F39C12"
ENTRY_COLOR = "#A0A0CC"
TARGET_COLOR = "#2ECC71"
STOP_COLOR = "#E74C3C"
BAND_COLOR = "rgba(160,160,204,0.15)"
GRID_COLOR = "rgba(160,160,204,0.12)"
TEXT_COLOR = "#E0E0F0"


def _parse_strat(top_rec_json: str | None) -> dict:
    if not top_rec_json:
        return {}
    try:
        return json.loads(top_rec_json)
    except Exception:
        return {}


def _parse_indicators(indicators_json: str | None) -> dict:
    if not indicators_json:
        return {}
    try:
        return json.loads(indicators_json)
    except Exception:
        return {}


def build_pulse_chart(pulse: dict, lookback_days: int = 90) -> go.Figure | None:
    """Build an interactive Plotly chart for a single pulse row.

    Returns ``None`` if we can't fetch price data for the ticker.
    """
    ticker = pulse.get("ticker") or ""
    entry_price = float(pulse.get("entry_price") or 0)
    entry_date_str = pulse.get("entry_date") or ""
    expiry_str = pulse.get("expiry") or ""
    sigma = float(pulse.get("sigma") or 0.22)
    outcome = pulse.get("outcome")
    outcome_price = pulse.get("outcome_price")
    outcome_pnl_pct = pulse.get("outcome_pnl_pct")
    status = pulse.get("status") or "active"
    strat = _parse_strat(pulse.get("top_rec_json"))
    indicators = _parse_indicators(pulse.get("indicators_json"))

    target_price = strat.get("target_price") or strat.get("strike")
    stop_price = strat.get("stop_price") or strat.get("breakeven_lower")
    strategy_label = pulse.get("strategy_label") or strat.get("strategy_type") or ""

    # Parse dates
    try:
        entry_dt = datetime.fromisoformat(entry_date_str).date() if entry_date_str else date.today()
    except Exception:
        entry_dt = date.today()
    try:
        expiry_dt = datetime.fromisoformat(expiry_str).date() if expiry_str else entry_dt + timedelta(days=21)
    except Exception:
        expiry_dt = entry_dt + timedelta(days=21)

    # Fetch bars: lookback + forward from entry to today (or expiry)
    total_days = (date.today() - entry_dt).days + lookback_days + 10
    bars = fetch_bars(ticker, max(total_days, lookback_days + 30))
    if not bars or not bars.get("close") or len(bars["close"]) < 10:
        return None

    closes = bars["close"]
    n = len(closes)
    bar_dates = [date.today() - timedelta(days=n - 1 - i) for i in range(n)]

    # Slice: lookback_days before entry through latest
    try:
        entry_idx = min(range(n), key=lambda i: abs((bar_dates[i] - entry_dt).days))
    except Exception:
        entry_idx = max(0, n - 30)
    start_idx = max(0, entry_idx - lookback_days)
    chart_dates = bar_dates[start_idx:]
    chart_closes = closes[start_idx:]
    latest_price = chart_closes[-1] if chart_closes else entry_price

    # ── Expected-move bands (forward from entry to expiry) ───────────────
    band_dates: list[date] = []
    band_upper: list[float] = []
    band_lower: list[float] = []
    dte_total = max((expiry_dt - entry_dt).days, 1)
    for d in chart_dates:
        if d < entry_dt:
            continue
        t_days = (d - entry_dt).days
        t_frac = t_days / 252.0
        move = entry_price * sigma * math.sqrt(max(t_frac, 0.0001))
        band_dates.append(d)
        band_upper.append(entry_price + move)
        band_lower.append(entry_price - move)

    # ── P&L computation ──────────────────────────────────────────────────
    if outcome_pnl_pct is not None:
        pnl_pct = float(outcome_pnl_pct)
    elif entry_price > 0:
        pnl_pct = (latest_price - entry_price) / entry_price * 100
    else:
        pnl_pct = 0.0

    pnl_color = WIN if pnl_pct > 0 else LOSS if pnl_pct < 0 else NEUTRAL

    # ── Build figure ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Expected-move cone (filled area)
    if band_dates:
        fig.add_trace(go.Scatter(
            x=band_dates, y=band_upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=band_dates, y=band_lower,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=BAND_COLOR,
            name="Expected move (±1σ)",
            hoverinfo="skip",
        ))

    # Price line
    fig.add_trace(go.Scatter(
        x=chart_dates, y=chart_closes,
        mode="lines",
        line=dict(color=pnl_color, width=2.5),
        name=f"{ticker} price",
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ))

    # Reference lines — use add_shape (not add_hline/vline which break
    # on mixed date/numeric axes in older Plotly versions).
    y_min = min(chart_closes) * 0.97
    y_max = max(chart_closes) * 1.03
    x_min_str = chart_dates[0].isoformat()
    x_max_str = chart_dates[-1].isoformat()

    def _hline(y: float, color: str, dash: str, label: str, pos: str = "left"):
        fig.add_shape(type="line", x0=x_min_str, x1=x_max_str, y0=y, y1=y,
                      line=dict(color=color, width=1, dash=dash))
        fig.add_annotation(
            x=x_min_str if pos == "left" else x_max_str,
            y=y, text=label, showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left" if pos == "left" else "right",
            yshift=10,
        )

    def _vline(d: date, color: str, dash: str, label: str):
        ds = d.isoformat()
        fig.add_shape(type="line", x0=ds, x1=ds, y0=y_min, y1=y_max,
                      line=dict(color=color, width=1, dash=dash))
        fig.add_annotation(
            x=ds, y=y_max, text=label, showarrow=False,
            font=dict(color=color, size=10), yshift=8,
        )

    if entry_price > 0:
        _hline(entry_price, ENTRY_COLOR, "dash", f"Entry ${entry_price:.2f}")
    if target_price and float(target_price) > 0:
        _hline(float(target_price), TARGET_COLOR, "dot", f"Target ${float(target_price):.2f}", "right")
    if stop_price and float(stop_price) > 0:
        _hline(float(stop_price), STOP_COLOR, "dot", f"Stop ${float(stop_price):.2f}", "right")

    _vline(entry_dt, ENTRY_COLOR, "dash", "Entry")
    if expiry_dt > entry_dt:
        _vline(expiry_dt, "rgba(160,160,204,0.4)", "dot", "Expiry")

    # Outcome marker
    if outcome and outcome != "neutral" and outcome_price:
        marker_color = WIN if outcome == "win" else LOSS
        marker_symbol = "star" if outcome == "win" else "x"
        fig.add_trace(go.Scatter(
            x=[chart_dates[-1]], y=[float(outcome_price)],
            mode="markers+text",
            marker=dict(size=14, color=marker_color, symbol=marker_symbol),
            text=[f"{'✓' if outcome == 'win' else '✗'} {pnl_pct:+.1f}%"],
            textposition="top center",
            textfont=dict(color=marker_color, size=13),
            showlegend=False,
            hovertemplate=f"Outcome: {outcome}<br>${float(outcome_price):.2f}<br>{pnl_pct:+.1f}%<extra></extra>",
        ))

    # P&L badge (top-right annotation)
    badge = f"{pnl_pct:+.1f}%"
    badge_label = status.upper() if status == "active" else outcome.upper() if outcome else status.upper()

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        text=f"<b>{badge_label}</b>  <b>{badge}</b>",
        font=dict(size=16, color=pnl_color),
        showarrow=False, align="right",
        bgcolor="rgba(8,8,26,0.7)", borderpad=6,
    )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        title=dict(
            text=f"{ticker} · {strategy_label}",
            font=dict(color=TEXT_COLOR, size=15),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=10),
        ),
        yaxis=dict(
            title="Price ($)",
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=10),
            title_font=dict(color=TEXT_COLOR, size=11),
            tickprefix="$",
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        height=380,
        hovermode="x unified",
    )

    return fig
