"""Execute paper trades on Alpaca based on live pulse signals.

MVP: stock market orders only (no multi-leg options — those need an
options-approved paper account). Direction is derived from PGI sign:
  PGI ≥ 10  → BUY  (bullish signal)
  PGI ≤ -10 → SELL (close existing long, or skip if no position)
  |PGI| < 10 → WATCH (no trade)

Each order is tagged ``client_order_id = "{algo_id}_{pulse_id}"`` so
the existing ``algo_from_client_order_id`` helper attributes fills back
to algos on the /Orders page.

Safety:
  * Default trade size: $500 per signal (configurable per-algo via
    ``execution_size_usd`` in coefficients).
  * Max open positions per account: 10 (configurable).
  * Execution is disabled by default — flip ``execution_enabled: true``
    in algo coefficients on /Admin to arm.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from . import store
from .settings import AccountConfig, load_accounts

log = logging.getLogger(__name__)

DEFAULT_SIZE_USD = 500.0
DEFAULT_MAX_POSITIONS = 10


@dataclass
class OrderResult:
    success: bool
    order_id: str | None = None
    client_order_id: str | None = None
    side: str | None = None
    qty: int = 0
    symbol: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "side": self.side,
            "qty": self.qty,
            "symbol": self.symbol,
            "error": self.error,
        }


def _account_for_algo(algo_id: str) -> AccountConfig | None:
    accounts = load_accounts()
    for acc in accounts:
        if algo_id in acc.algos and acc.is_configured:
            return acc
    return None


def _client_for_algo(algo_id: str) -> tuple[TradingClient, AccountConfig] | None:
    acc = _account_for_algo(algo_id)
    if not acc:
        return None
    return TradingClient(api_key=acc.key, secret_key=acc.secret, paper=True), acc


def is_execution_enabled(algo_id: str) -> bool:
    coefs = store.get_coefficients(algo_id)
    return bool(coefs.get("execution_enabled", False))


def execute_pulse(
    algo_id: str,
    pulse_id: str,
    ticker: str,
    pgi: float,
    entry_price: float,
    strategy_type: str | None = None,
) -> OrderResult:
    """Submit a paper order for a live pulse signal.

    Returns an ``OrderResult`` with the order_id on success, or an error
    message on failure. Never raises — errors are returned, not thrown.
    """
    # Direction from PGI
    if pgi >= 10:
        side = OrderSide.BUY
    elif pgi <= -10:
        side = OrderSide.SELL
    else:
        return OrderResult(success=False, error="PGI too neutral for execution")

    # Get trading client
    pair = _client_for_algo(algo_id)
    if not pair:
        return OrderResult(success=False, error=f"no configured account for {algo_id}")
    client, acc = pair

    try:
        # Check execution budget
        coefs = store.get_coefficients(algo_id)
        size_usd = float(coefs.get("execution_size_usd", DEFAULT_SIZE_USD))
        max_pos = int(coefs.get("max_positions", DEFAULT_MAX_POSITIONS))

        # Guard: max open positions
        positions = client.get_all_positions()
        if len(positions) >= max_pos and side == OrderSide.BUY:
            return OrderResult(
                success=False,
                error=f"max positions reached ({len(positions)}/{max_pos})",
            )

        # For SELL: only sell if we actually hold the ticker
        if side == OrderSide.SELL:
            held = [p for p in positions if p.symbol == ticker]
            if not held:
                return OrderResult(
                    success=False, error=f"no position in {ticker} to sell",
                )
            # Sell entire position
            qty = int(float(held[0].qty))
            if qty <= 0:
                return OrderResult(success=False, error=f"zero qty for {ticker}")
        else:
            # BUY: calculate qty from budget
            if entry_price <= 0:
                return OrderResult(success=False, error="entry_price <= 0")
            qty = max(1, math.floor(size_usd / entry_price))

        client_order_id = f"{algo_id}_{pulse_id}"
        if len(client_order_id) > 48:
            client_order_id = client_order_id[:48]

        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = client.submit_order(req)

        # Persist to our orders table
        store.save_order({
            "order_id": str(order.id),
            "pulse_id": pulse_id,
            "algo_id": algo_id,
            "ticker": ticker,
            "side": side.value,
            "qty": qty,
            "status": str(order.status),
            "client_order_id": client_order_id,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
        })

        log.info(f"order submitted: {side.value} {qty}× {ticker} for {algo_id} "
                 f"(${size_usd:.0f} budget, order={order.id})")

        return OrderResult(
            success=True,
            order_id=str(order.id),
            client_order_id=client_order_id,
            side=side.value,
            qty=qty,
            symbol=ticker,
        )

    except Exception as e:  # noqa: BLE001
        log.warning(f"order execution failed for {algo_id}/{ticker}: {e}")
        return OrderResult(success=False, error=f"{type(e).__name__}: {e}")


def get_account_positions(algo_id: str) -> list[dict]:
    """Return current positions for an algo's paper account."""
    pair = _client_for_algo(algo_id)
    if not pair:
        return []
    client, _ = pair
    try:
        return [p.model_dump() for p in client.get_all_positions()]
    except Exception:  # noqa: BLE001
        return []


def get_account_summary(algo_id: str) -> dict | None:
    """Return equity/cash/P&L for an algo's paper account."""
    pair = _client_for_algo(algo_id)
    if not pair:
        return None
    client, acc = pair
    try:
        a = client.get_account()
        equity = float(a.equity)
        last_equity = float(a.last_equity) if a.last_equity else equity
        return {
            "algo_id": algo_id,
            "account": acc.name,
            "equity": equity,
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "pnl_today": equity - last_equity,
            "pnl_today_pct": ((equity - last_equity) / last_equity * 100) if last_equity else 0,
            "positions": len(client.get_all_positions()),
        }
    except Exception:  # noqa: BLE001
        return None
