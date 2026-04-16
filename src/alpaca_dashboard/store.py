"""Backtest data store — local SQLite by default, Turso (libsql) when
``TURSO_DATABASE_URL`` is set.

Schema:
  pulses            — one row per simulated trade outcome
  coefficients      — algo-level tunable knobs (overrides from /admin)
  jobs              — admin-triggered backtest runs (status + summary)

Both drivers are DB-API 2.0 compatible; we avoid ``sqlite3.Row`` so row access
looks the same on either backend (``_rows``/``_row`` helpers zip column names
from ``cursor.description``).
"""
from __future__ import annotations

# Version marker — bump any time this module changes in a way where a
# warm Streamlit Cloud process might still have a stale cached copy.
# Cloud's hot-reload only touches files present in the commit diff, so
# cross-file additions (e.g. new helpers used only by downstream pages)
# can go unnoticed. The marker keeps this file in every such diff.
STORE_VERSION = "2026-04-16.1"

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable

from .settings import db_path

log = logging.getLogger(__name__)


def _turso_url() -> str:
    return (os.getenv("TURSO_DATABASE_URL") or "").strip()


def _turso_token() -> str:
    return (os.getenv("TURSO_AUTH_TOKEN") or "").strip()


def using_turso() -> bool:
    return bool(_turso_url())


# ── Turso HTTP-pipeline driver ────────────────────────────────────────────────
#
# We talk to Turso directly over its public ``POST /v2/pipeline`` endpoint
# (Hrana-over-HTTP). This avoids the libsql-experimental Rust/cmake build
# (no wheel on Streamlit Cloud) and the libsql-client sync wrapper's
# event-loop quirks. The only runtime dep is ``requests``.

_TURSO_TYPE_MAP = {int: "integer", float: "float", str: "text", bytes: "blob"}


def _encode_arg(v: Any) -> dict:
    if v is None:
        return {"type": "null", "value": None}
    if isinstance(v, bool):
        return {"type": "integer", "value": "1" if v else "0"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _decode_val(cell: dict) -> Any:
    t = cell.get("type")
    v = cell.get("value")
    if t == "null":
        return None
    if t == "integer":
        return int(v) if v is not None else None
    if t == "float":
        return float(v) if v is not None else None
    return v


def _turso_execute(sql: str, params=()) -> dict:
    import requests  # lazy import

    url = _turso_url().replace("libsql://", "https://", 1).rstrip("/") + "/v2/pipeline"
    token = _turso_token()
    body = {
        "requests": [
            {"type": "execute", "stmt": {
                "sql": sql,
                "args": [_encode_arg(p) for p in (params or [])],
            }},
            {"type": "close"},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.post(url, json=body, headers=headers, timeout=30)
    if not r.ok:
        # Surface Turso's actual error body — r.raise_for_status() eats it
        raise RuntimeError(
            f"Turso HTTP {r.status_code}: {r.text[:500]} "
            f"(url={url!r}, token_len={len(token)}, sql_preview={sql[:100]!r})"
        )
    data = r.json()
    first = data["results"][0]
    if first.get("type") == "error":
        err = first.get("error", {})
        raise RuntimeError(f"Turso error: {err.get('message') or err}")
    return first["response"]["result"]


class _TursoCursor:
    """sqlite3-cursor-like view over a Turso pipeline ``execute`` result."""

    def __init__(self, result: dict):
        cols = result.get("cols") or []
        self.description = [(c.get("name"),) for c in cols] if cols else None
        self.rowcount = int(result.get("affected_row_count") or 0)
        self._rows = [
            tuple(_decode_val(cell) for cell in row)
            for row in (result.get("rows") or [])
        ]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _TursoConn:
    """sqlite3-connection-like wrapper; each ``execute`` opens one pipeline."""

    def execute(self, sql: str, params=()):
        return _TursoCursor(_turso_execute(sql, params))

    def commit(self):
        pass  # Turso pipeline auto-commits each execute

    def close(self):
        pass


def _connect():
    """Return a DB-API-compatible connection for the active backend."""
    if using_turso():
        return _TursoConn()

    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _rows(cur) -> list[dict]:
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur) -> dict | None:
    cols = [d[0] for d in cur.description] if cur.description else []
    r = cur.fetchone()
    return dict(zip(cols, r)) if r else None


@contextmanager
def cursor():
    conn = _connect()
    try:
        yield conn
        # libsql requires explicit commit; local sqlite uses isolation_level=None
        # (auto-commit) but a second commit() is a harmless no-op.
        try:
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS pulses (
        pulse_id         TEXT PRIMARY KEY,
        algo_id          TEXT NOT NULL,
        ticker           TEXT NOT NULL,
        pulse_type       TEXT,
        strategy_label   TEXT,
        entry_date       TEXT NOT NULL,
        entry_price      REAL,
        expiry           TEXT,
        dte              INTEGER,
        pgi              REAL,
        sigma            REAL,
        status           TEXT,
        outcome          TEXT,
        outcome_price    REAL,
        outcome_pnl_pct  REAL,
        selection_reason TEXT,
        job_id           TEXT,
        top_rec_json     TEXT,
        indicators_json  TEXT,
        market_regime    TEXT,
        cap_bucket       TEXT,
        created_at       TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coefficients (
        algo_id    TEXT PRIMARY KEY,
        payload    TEXT NOT NULL,
        updated_at TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_cursors (
        destination            TEXT PRIMARY KEY,
        last_pushed_created_at TEXT,
        last_pushed_pulse_id   TEXT,
        last_pushed_count      INTEGER DEFAULT 0,
        last_error             TEXT,
        updated_at             TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id     TEXT PRIMARY KEY,
        algo_id    TEXT,
        status     TEXT NOT NULL,
        started_at TEXT,
        ended_at   TEXT,
        params     TEXT,
        summary    TEXT,
        error      TEXT
    )
    """,
]


# Columns added to `pulses` in later migrations. Keep idempotent — each
# ALTER TABLE is wrapped in try/except because SQLite/libsql don't support
# `ADD COLUMN IF NOT EXISTS`.
_ADD_COLUMN_MIGRATIONS = [
    "ALTER TABLE pulses ADD COLUMN top_rec_json    TEXT",
    "ALTER TABLE pulses ADD COLUMN indicators_json TEXT",
    "ALTER TABLE pulses ADD COLUMN market_regime   TEXT",
    "ALTER TABLE pulses ADD COLUMN cap_bucket      TEXT",
    "ALTER TABLE pulses ADD COLUMN generated_at    TEXT",
    "ALTER TABLE ingestion_cursors ADD COLUMN last_pushed_pulse_id TEXT",
]

_SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pulses_algo   ON pulses(algo_id)",
    "CREATE INDEX IF NOT EXISTS idx_pulses_ticker ON pulses(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pulses_entry  ON pulses(entry_date)",
    "CREATE INDEX IF NOT EXISTS idx_pulses_job    ON pulses(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_pulses_regime ON pulses(market_regime)",
    "CREATE INDEX IF NOT EXISTS idx_pulses_cap    ON pulses(cap_bucket)",
]


def init_db() -> None:
    with cursor() as c:
        # 1. Tables (idempotent via IF NOT EXISTS).
        for stmt in _SCHEMA_TABLES:
            c.execute(stmt)
        # 2. Add-column migrations (idempotent via try/except — SQLite/libsql
        #    have no `ADD COLUMN IF NOT EXISTS`).
        for stmt in _ADD_COLUMN_MIGRATIONS:
            try:
                c.execute(stmt)
            except Exception:
                pass
        # 3. Indexes last — some reference newly-added columns.
        for stmt in _SCHEMA_INDEXES:
            c.execute(stmt)


# ── Pulses ────────────────────────────────────────────────────────────────────

_PULSE_COLS = [
    "pulse_id", "algo_id", "ticker", "pulse_type", "strategy_label",
    "entry_date", "entry_price", "expiry", "dte", "pgi", "sigma",
    "status", "outcome", "outcome_price", "outcome_pnl_pct",
    "selection_reason", "job_id",
    "top_rec_json", "indicators_json", "market_regime", "cap_bucket",
    "generated_at",
]


def save_pulse(row: dict[str, Any]) -> None:
    values = tuple(row.get(col) for col in _PULSE_COLS)
    placeholders = ",".join("?" * len(_PULSE_COLS))
    with cursor() as c:
        c.execute(
            f"INSERT OR REPLACE INTO pulses ({','.join(_PULSE_COLS)}) "
            f"VALUES ({placeholders})",
            values,
        )


def save_pulses(rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        save_pulse(r)
        n += 1
    return n


def pulses_for_algo(algo_id: str, limit: int = 5000) -> list[dict]:
    with cursor() as c:
        cur = c.execute(
            "SELECT * FROM pulses WHERE algo_id = ? ORDER BY entry_date DESC LIMIT ?",
            (algo_id, limit),
        )
        return _rows(cur)


def all_pulses(limit: int = 20000) -> list[dict]:
    with cursor() as c:
        cur = c.execute(
            "SELECT * FROM pulses ORDER BY entry_date DESC LIMIT ?",
            (limit,),
        )
        return _rows(cur)


def delete_pulses_for_algo(algo_id: str) -> int:
    with cursor() as c:
        cur = c.execute("DELETE FROM pulses WHERE algo_id = ?", (algo_id,))
        return cur.rowcount or 0


def delete_all_pulses() -> int:
    with cursor() as c:
        cur = c.execute("DELETE FROM pulses")
        return cur.rowcount or 0


def pulses_since(
    since_created_at: str | None = None,
    since_pulse_id: str | None = None,
    algo_id: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Paginated read for the ingestion pipeline.

    Uses a **composite cursor** ``(created_at, pulse_id)`` because
    ``created_at`` has second precision, so many pulses written in the
    same backtest batch share a timestamp. Sorting + filtering on
    ``(created_at, pulse_id)`` gives a total order so no row is skipped.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if since_created_at and since_pulse_id:
        clauses.append("(created_at > ? OR (created_at = ? AND pulse_id > ?))")
        params.extend([since_created_at, since_created_at, since_pulse_id])
    elif since_created_at:
        clauses.append("created_at > ?")
        params.append(since_created_at)
    if algo_id:
        clauses.append("algo_id = ?")
        params.append(algo_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with cursor() as c:
        cur = c.execute(
            f"SELECT * FROM pulses{where} "
            f"ORDER BY created_at ASC, pulse_id ASC LIMIT ?",
            tuple(params),
        )
        return _rows(cur)


# ── Ingestion cursors ────────────────────────────────────────────────────────

def get_ingestion_cursor(destination: str) -> dict | None:
    with cursor() as c:
        cur = c.execute(
            "SELECT * FROM ingestion_cursors WHERE destination = ?",
            (destination,),
        )
        return _row(cur)


def set_ingestion_cursor(
    destination: str,
    last_pushed_created_at: str,
    last_pushed_count: int,
    last_pushed_pulse_id: str | None = None,
    last_error: str | None = None,
) -> None:
    with cursor() as c:
        c.execute(
            """
            INSERT INTO ingestion_cursors(
                destination, last_pushed_created_at, last_pushed_pulse_id,
                last_pushed_count, last_error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(destination) DO UPDATE SET
                last_pushed_created_at = excluded.last_pushed_created_at,
                last_pushed_pulse_id   = excluded.last_pushed_pulse_id,
                last_pushed_count      = excluded.last_pushed_count,
                last_error             = excluded.last_error,
                updated_at             = excluded.updated_at
            """,
            (destination, last_pushed_created_at, last_pushed_pulse_id,
             last_pushed_count, last_error),
        )


# ── Coefficients ──────────────────────────────────────────────────────────────

def get_coefficients(algo_id: str) -> dict[str, Any]:
    with cursor() as c:
        cur = c.execute(
            "SELECT payload FROM coefficients WHERE algo_id = ?", (algo_id,)
        )
        row = _row(cur)
    return json.loads(row["payload"]) if row else {}


def set_coefficients(algo_id: str, payload: dict[str, Any]) -> None:
    with cursor() as c:
        c.execute(
            """
            INSERT INTO coefficients(algo_id, payload, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(algo_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (algo_id, json.dumps(payload)),
        )


# ── Jobs ──────────────────────────────────────────────────────────────────────

def create_job(job_id: str, algo_id: str | None, params: dict) -> None:
    with cursor() as c:
        c.execute(
            """
            INSERT INTO jobs(job_id, algo_id, status, started_at, params)
            VALUES (?, ?, 'running', ?, ?)
            """,
            (job_id, algo_id, datetime.utcnow().isoformat(), json.dumps(params)),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    params = tuple(list(fields.values()) + [job_id])
    with cursor() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", params)


def finish_job(job_id: str, summary: dict | None = None, error: str | None = None) -> None:
    update_job(
        job_id,
        status="error" if error else "done",
        ended_at=datetime.utcnow().isoformat(),
        summary=json.dumps(summary) if summary else None,
        error=error,
    )


def list_jobs(limit: int = 50) -> list[dict]:
    with cursor() as c:
        cur = c.execute(
            "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        return _rows(cur)


def get_job(job_id: str) -> dict | None:
    with cursor() as c:
        cur = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return _row(cur)
