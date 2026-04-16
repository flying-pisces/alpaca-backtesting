"""Destinations that consume market_pulse-shaped rows produced by converter.

Design: one thin abstract ``PulseDestination`` + two concrete adapters.
Swap destinations without changing pipeline code.

- ``SqliteDestination`` — writes directly into a market_pulse SQLite file
  (what we use today when the file is on the same host, e.g. local dev or
  a shared volume on Fly.io).
- ``HttpDestination`` — stub for the future, when market_pulse grows an
  ingestion endpoint. Contract is already defined; implementation is a
  ~30-line ``requests.post`` once the endpoint exists.
"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .converter import TARGET_COLUMNS


class PulseDestination(ABC):
    """One method: write a batch of market_pulse-shaped rows, return count written.

    Implementations must be idempotent on ``pulse_id`` (caller may retry the
    same batch on transient failure).
    """

    name: str = "abstract"

    @abstractmethod
    def write(self, rows: Iterable[dict[str, Any]]) -> int:
        raise NotImplementedError

    def healthcheck(self) -> tuple[bool, str]:
        """Cheap verification that the destination is reachable/writable."""
        return True, "ok"


# ── SqliteDestination ────────────────────────────────────────────────────────

class SqliteDestination(PulseDestination):
    """Writes into a market_pulse ``pulses.db`` SQLite file via direct DB-API.

    Schema match is enforced by ``TARGET_COLUMNS``. The target table is
    expected to already exist (market_pulse creates it on first boot). If
    it doesn't, the first ``write()`` call raises; we don't auto-create
    because the schema is market_pulse's contract, not ours.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.name = f"sqlite:{self.db_path}"

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"market_pulse db not found at {self.db_path}. "
                f"Start market_pulse once to create the file, or point at "
                f"the correct path."
            )
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def healthcheck(self) -> tuple[bool, str]:
        try:
            conn = self._connect()
        except FileNotFoundError as e:
            return False, str(e)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master "
                               "WHERE type='table' AND name='pulses'")
            if cur.fetchone() is None:
                return False, "pulses table missing (market_pulse not initialised yet)"
            # Verify every column we want to INSERT exists. If the DB is
            # pre-migration, INSERT will fail with a cryptic error later —
            # surface a concrete "run market_pulse once" message up front.
            have = {r[1] for r in conn.execute("PRAGMA table_info(pulses)").fetchall()}
            missing = [c for c in TARGET_COLUMNS if c not in have]
            if missing:
                return False, (
                    f"market_pulse pulses table is missing columns "
                    f"{missing[:6]}{'…' if len(missing) > 6 else ''}. "
                    f"Start market_pulse once so it can run its ALTER TABLE "
                    f"migrations, then retry."
                )
            return True, f"ok — {self.db_path} ({len(have)} cols)"
        finally:
            conn.close()

    def write(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        placeholders = ",".join("?" * len(TARGET_COLUMNS))
        sql = (
            f"INSERT OR IGNORE INTO pulses ({','.join(TARGET_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        data = [tuple(r.get(c) for c in TARGET_COLUMNS) for r in rows]
        # market_pulse's server may hold a write lock while it's running.
        # SQLite in WAL mode allows concurrent readers but serialises writes.
        # We retry with backoff to wait for a short lock gap.
        import time
        last_err = None
        for attempt in range(5):
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.executemany(sql, data)
                conn.commit()
                return cur.rowcount if cur.rowcount >= 0 else len(rows)
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" in str(e).lower() and attempt < 4:
                    time.sleep(2 ** attempt)   # 1, 2, 4, 8s backoff
                    continue
                raise
            finally:
                conn.close()
        raise last_err  # type: ignore[misc]


# ── HttpDestination ──────────────────────────────────────────────────────────

MAX_HTTP_BATCH = 500   # server-side cap per request


class HttpDestination(PulseDestination):
    """POST a batch of pulses to market_pulse's ``/api/ingest-pulses``.

    The endpoint lives at ``signalpro-pulse.fly.dev`` and is authenticated
    with a service-to-service Bearer token (``INGEST_API_KEY`` on the
    server side — NOT a user JWT). Request body is
    ``{"pulses": [{...}, ...]}``, max 500 per request. Response is
    ``{"inserted": N, "skipped": M, "total": N+M}``.

    Reference: ``ref/market_pulse/backend/server.py:10127-10182``.
    """

    def __init__(
        self,
        base_url: str = "https://signalpro-pulse.fly.dev",
        auth_token: str | None = None,
        *,
        path: str = "/api/ingest-pulses",
    ):
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.auth_token = auth_token or ""
        self.name = f"http:{self.base_url}{self.path}"
        self._url = f"{self.base_url}{self.path}"
        self._health_url = f"{self._url}/health"

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def healthcheck(self) -> tuple[bool, str]:
        import requests
        if not self.auth_token:
            return False, "INGEST_API_KEY not set"
        try:
            r = requests.get(self._health_url, headers=self._headers(), timeout=10)
            if r.ok:
                return True, f"ok — {self._url}"
            return False, f"health {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, f"unreachable: {e}"

    def write(self, rows: Iterable[dict[str, Any]]) -> int:
        import requests
        rows = list(rows)
        if not rows:
            return 0
        total_inserted = 0
        # Respect the server's 500-per-request cap.
        for i in range(0, len(rows), MAX_HTTP_BATCH):
            batch = rows[i : i + MAX_HTTP_BATCH]
            r = requests.post(
                self._url,
                json={"pulses": batch},
                headers=self._headers(),
                timeout=30,
            )
            if not r.ok:
                raise RuntimeError(
                    f"ingest-pulses {r.status_code}: {r.text[:300]}"
                )
            body = r.json()
            total_inserted += body.get("inserted", 0)
        return total_inserted
