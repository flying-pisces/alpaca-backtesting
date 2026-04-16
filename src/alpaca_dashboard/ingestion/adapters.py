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


# ── HttpDestination (stub) ───────────────────────────────────────────────────

class HttpDestination(PulseDestination):
    """Post a batch to a future market_pulse ingestion endpoint.

    market_pulse does not expose one today. When it does, implement this
    class as a thin ``requests.post`` with Bearer auth (market_pulse uses
    JWT — see ``backend/auth.py:35-45`` in their repo). The pipeline code
    does not need to change.
    """

    def __init__(self, base_url: str, auth_token: str, *,
                 path: str = "/api/ingest-pulses"):
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.auth_token = auth_token
        self.name = f"http:{self.base_url}{self.path}"

    def healthcheck(self) -> tuple[bool, str]:
        return False, "HttpDestination not implemented yet (waiting on market_pulse endpoint)"

    def write(self, rows: Iterable[dict[str, Any]]) -> int:
        raise NotImplementedError(
            "HttpDestination.write is a stub. market_pulse needs an "
            "/api/ingest-pulses endpoint before this can be wired up. "
            "See ref/market_pulse/backend/server.py; pattern similar to "
            "/api/save-ai-pulse (line 5412) with batch support."
        )
