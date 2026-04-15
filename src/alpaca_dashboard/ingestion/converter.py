"""Shape our backtest pulse row into the schema market_pulse expects.

Reference for the target schema: ``backend/db/models.py:13-33`` in the
sibling ``market_pulse`` repo (copy kept at ``ref/market_pulse/`` which
is gitignored).

Key field mappings documented inline below. Unknown/optional fields are
left ``None`` rather than fabricated, so a reader can always tell "this
came from backtest, we don't have that data" from "this is a live pulse".
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

# Per `backend/server.py:2487` in market_pulse — risk level is an int 1–5.
_RISK_LEVEL_BY_ALGO = {
    "degen": 4,
    "surge": 3,
    "moderate": 2,
    "sentinel": 2,
    "fortress": 1,
    # planned algos — sensible defaults if they ever flow through here
    "reddit_play": 4,
    "er_play": 3,
    "dividend_play": 1,
}

# Duration bucket used by market_pulse UI to group pulses on the timeline.
_DURATION_BY_ALGO = {
    "degen": "short",
    "surge": "short",
    "moderate": "medium",
    "sentinel": "long",
    "fortress": "long",
    "reddit_play": "short",
    "er_play": "short",
    "dividend_play": "long",
}

INGESTION_VERSION = "alpaca_backtesting_v1"


def _score_from_pgi(pgi: float | None) -> int:
    """Match market_pulse's existing convention (server.py:2487-ish).

    Maps PGI ∈ [-100, +100] into a 30–95 score band. 0 PGI → 50.
    """
    if pgi is None:
        return 50
    return int(max(30, min(95, 50 + pgi / 2)))


def _action_from_pgi(pgi: float | None) -> str:
    """BUY when bullish signal, SELL when bearish, WATCH when neutral."""
    if pgi is None or abs(pgi) < 10:
        return "WATCH"
    return "BUY" if pgi >= 0 else "SELL"


def _strategy_fields(top_rec_json: str | None) -> dict[str, Any]:
    """Pull target_price/stop_price/strike out of a serialised strategy dict."""
    if not top_rec_json:
        return {}
    try:
        strat = json.loads(top_rec_json)
    except Exception:
        return {}
    return {
        "target_price": strat.get("target_price") or strat.get("strike"),
        "stop_price": strat.get("stop_price") or strat.get("breakeven_lower"),
    }


def to_market_pulse_row(our_row: dict[str, Any]) -> dict[str, Any]:
    """Convert one of our backtest rows into a market_pulse-shaped pulse dict.

    Output is a flat dict whose keys correspond 1:1 to columns in the
    ``pulses`` table in market_pulse. Callers are expected to ``INSERT OR
    IGNORE`` on ``pulse_id`` for idempotency.
    """
    algo_id = our_row.get("algo_id") or "unknown"
    pgi = our_row.get("pgi")
    strat = _strategy_fields(our_row.get("top_rec_json"))

    entry_date = our_row.get("entry_date")
    if entry_date:
        generated_at = f"{entry_date}T00:00:00Z"
    else:
        generated_at = datetime.utcnow().isoformat() + "Z"

    # Resolve the expiry timestamp into a market_pulse-compatible iso-Z form.
    expiry = our_row.get("expiry")
    expires_at = f"{expiry}T16:00:00Z" if expiry else None

    # Backtest pulses are fully resolved at write time, so resolved_at =
    # entry_date + dte (caps the holding period). For still-active pulses
    # leave null.
    resolved_at = None
    if our_row.get("status") and our_row.get("status") != "active":
        try:
            d = datetime.fromisoformat(entry_date) + timedelta(days=int(our_row.get("dte") or 0))
            resolved_at = d.isoformat() + "Z"
        except Exception:
            resolved_at = None

    outcome_pnl = None
    op, ep = our_row.get("outcome_price"), our_row.get("entry_price")
    if op is not None and ep is not None:
        outcome_pnl = round(float(op) - float(ep), 2)

    # ``scan_json`` — reserved by market_pulse for raw scan metadata. We
    # pass through our indicators snapshot so a downstream reader can see
    # the exact PGI/RSI/SMA values that drove the signal.
    scan_json = our_row.get("indicators_json")

    return {
        # Identity
        "pulse_id": our_row["pulse_id"],
        "pulse_type": our_row.get("pulse_type"),
        "ticker": our_row.get("ticker"),
        # Provenance
        "generated_at": generated_at,
        "generated_by": INGESTION_VERSION,
        "algo_id": algo_id,
        "algo_version": "backtest_algo_v2",
        # Classification
        "risk_level": _RISK_LEVEL_BY_ALGO.get(algo_id, 2),
        "score": _score_from_pgi(pgi),
        "action": _action_from_pgi(pgi),
        "asset_class": "equities",
        "duration_type": _DURATION_BY_ALGO.get(algo_id, "medium"),
        # Narrative
        "thesis": (our_row.get("selection_reason") or "")[:500],
        "summary": (our_row.get("selection_reason") or "")[:200],
        # Lifecycle
        "status": our_row.get("status"),
        "expires_at": expires_at,
        "resolved_at": resolved_at,
        # Prices
        "entry_price": our_row.get("entry_price"),
        "target_price": strat.get("target_price"),
        "stop_price": strat.get("stop_price"),
        # Outcome
        "outcome": our_row.get("outcome"),
        "outcome_price": our_row.get("outcome_price"),
        "outcome_pnl": outcome_pnl,
        "outcome_pnl_pct": our_row.get("outcome_pnl_pct"),
        "outcome_detail": (our_row.get("selection_reason") or "")[:200],
        "close_trigger": "backtest",
        # Blobs
        "scan_json": scan_json,
        "top_rec_json": our_row.get("top_rec_json"),
    }


# All columns market_pulse's ``pulses`` table accepts (minus the auto ``id``
# and ``created_at`` that it populates itself). The destination adapter uses
# this to build an INSERT statement.
TARGET_COLUMNS: tuple[str, ...] = (
    "pulse_id", "pulse_type", "ticker", "generated_at", "generated_by",
    "risk_level", "score", "action", "thesis", "summary", "status",
    "expires_at", "target_price", "stop_price", "resolved_at",
    "scan_json", "top_rec_json", "asset_class", "entry_price",
    "duration_type", "algo_version", "algo_id",
    "outcome", "outcome_price", "outcome_pnl", "outcome_pnl_pct",
    "outcome_detail", "close_trigger",
)
