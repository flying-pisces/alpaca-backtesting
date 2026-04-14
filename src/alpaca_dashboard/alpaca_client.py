"""Thin multi-account wrapper around alpaca-py.

Each configured account gets its own TradingClient. A `client_order_id` prefix
convention (e.g. ``degen_<uuid>``) is used to attribute fills back to an algo.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from .settings import AccountConfig


@dataclass
class AccountSnapshot:
    id: str
    name: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    positions_count: int
    pct_change_today: float


class AlpacaMultiClient:
    """Holds one TradingClient per configured account."""

    def __init__(self, accounts: Iterable[AccountConfig]):
        self._accounts = {a.id: a for a in accounts}

    @cached_property
    def clients(self) -> dict[str, TradingClient]:
        out: dict[str, TradingClient] = {}
        for acc in self._accounts.values():
            if not acc.is_configured:
                continue
            out[acc.id] = TradingClient(
                api_key=acc.key,
                secret_key=acc.secret,
                paper=True,
            )
        return out

    def account(self, account_id: str) -> AccountConfig:
        return self._accounts[account_id]

    def snapshot(self, account_id: str) -> AccountSnapshot | None:
        client = self.clients.get(account_id)
        if not client:
            return None
        acc = client.get_account()
        positions = client.get_all_positions()
        equity = float(acc.equity)
        last_equity = float(acc.last_equity) if acc.last_equity else equity
        pct = ((equity - last_equity) / last_equity * 100) if last_equity else 0.0
        return AccountSnapshot(
            id=account_id,
            name=self._accounts[account_id].name,
            equity=equity,
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            portfolio_value=float(acc.portfolio_value),
            status=str(acc.status),
            positions_count=len(positions),
            pct_change_today=pct,
        )

    def positions(self, account_id: str) -> list[dict]:
        client = self.clients.get(account_id)
        if not client:
            return []
        return [p.model_dump() for p in client.get_all_positions()]

    def orders(self, account_id: str, status: str = "all", limit: int = 200) -> list[dict]:
        client = self.clients.get(account_id)
        if not client:
            return []
        req = GetOrdersRequest(
            status=QueryOrderStatus(status),
            limit=limit,
            nested=True,
        )
        return [o.model_dump() for o in client.get_orders(filter=req)]

    @staticmethod
    def algo_from_client_order_id(client_order_id: str | None) -> str | None:
        """``degen_abc123`` → ``degen``."""
        if not client_order_id or "_" not in client_order_id:
            return None
        return client_order_id.split("_", 1)[0].lower()
