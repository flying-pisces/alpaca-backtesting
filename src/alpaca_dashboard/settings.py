"""Load config YAMLs + .env into typed structures."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class AccountConfig:
    id: str
    name: str
    description: str
    env_prefix: str
    algos: list[str]

    @property
    def key(self) -> str | None:
        return os.getenv(f"{self.env_prefix}_KEY") or None

    @property
    def secret(self) -> str | None:
        return os.getenv(f"{self.env_prefix}_SECRET") or None

    @property
    def account_id(self) -> str | None:
        return os.getenv(f"{self.env_prefix}_ID") or None

    @property
    def is_configured(self) -> bool:
        return bool(self.key and self.secret)


@dataclass(frozen=True)
class AlgoConfig:
    id: str
    name: str
    emoji: str
    risk: str
    sharpe_target: float
    dte_min: int
    dte_max: int
    pop_range: tuple[float, float]
    position_size_pct: float
    universe: Any
    pgi_gate: str
    strategies: list[str]
    enabled: bool
    status: str = "ready"     # "ready" | "planned"

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


def load_algos() -> list[AlgoConfig]:
    raw = yaml.safe_load((CONFIG_DIR / "algos.yaml").read_text())
    return [
        AlgoConfig(
            id=a["id"],
            name=a["name"],
            emoji=a["emoji"],
            risk=a["risk"],
            sharpe_target=float(a["sharpe_target"]),
            dte_min=int(a["dte_min"]),
            dte_max=int(a["dte_max"]),
            pop_range=tuple(a["pop_range"]),
            position_size_pct=float(a["position_size_pct"]),
            universe=a["universe"],
            pgi_gate=a["pgi_gate"],
            strategies=list(a["strategies"]),
            enabled=bool(a.get("enabled", True)),
            status=str(a.get("status", "ready")),
        )
        for a in raw["algos"]
    ]


def load_accounts() -> list[AccountConfig]:
    raw = yaml.safe_load((CONFIG_DIR / "accounts.yaml").read_text())
    return [
        AccountConfig(
            id=a["id"],
            name=a["name"],
            description=a["description"],
            env_prefix=a["env_prefix"],
            algos=list(a["algos"]),
        )
        for a in raw["accounts"]
    ]


def algo_to_account(accounts: list[AccountConfig]) -> dict[str, AccountConfig]:
    return {algo_id: acc for acc in accounts for algo_id in acc.algos}


def db_path() -> Path:
    return ROOT / os.getenv("DASHBOARD_DB_PATH", "data/trades.db")
