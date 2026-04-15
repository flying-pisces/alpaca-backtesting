"""Orchestrator: stream our backtest pulses into a ``PulseDestination``.

Scalability traits:
  * Paginated read from our Turso store via ``store.pulses_since(cursor)``
    so memory stays flat even for 100k+ pulses.
  * Idempotent writes — destinations use ``INSERT OR IGNORE`` on
    ``pulse_id``; retrying the same batch is safe.
  * Cursor persistence per-destination via ``ingestion_cursors`` table;
    ``push`` is safely resumable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import store
from .adapters import PulseDestination
from .converter import to_market_pulse_row

log = logging.getLogger(__name__)


@dataclass
class PushResult:
    destination: str
    total_read: int = 0
    total_written: int = 0
    batches: int = 0
    cursor_start: str | None = None
    cursor_end: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None
    per_algo: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "destination": self.destination,
            "total_read": self.total_read,
            "total_written": self.total_written,
            "batches": self.batches,
            "cursor_start": self.cursor_start,
            "cursor_end": self.cursor_end,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "error": self.error,
            "per_algo": self.per_algo,
        }


def push(
    destination: PulseDestination,
    *,
    since: str | None = None,
    algo_id: str | None = None,
    batch_size: int = 100,
    max_batches: int = 1000,
    reset_cursor: bool = False,
    progress_cb: Callable[[PushResult], None] | None = None,
) -> PushResult:
    """Push new pulses to ``destination``, advancing the per-destination cursor.

    Args:
        destination: where to write (SqliteDestination / HttpDestination)
        since: override the stored cursor with this ``created_at`` value. When
            ``None`` (default), resume from wherever the last ``push`` stopped.
        algo_id: filter to a single algo (useful for targeted re-pushes).
        batch_size: rows per destination write.
        max_batches: safety cap to avoid infinite loops.
        reset_cursor: start from the beginning of the table.
        progress_cb: called with the running ``PushResult`` after each batch.

    The cursor is ``pulses.created_at`` (ISO string). We order ASC so
    restarts resume cleanly.
    """
    t0 = time.time()
    result = PushResult(destination=destination.name)

    # Cursor resolution — composite (created_at, pulse_id) so we never skip
    # rows that share a created_at timestamp.
    if reset_cursor:
        cursor_ts, cursor_pid = None, None
    elif since is not None:
        cursor_ts, cursor_pid = since, None
    else:
        existing = store.get_ingestion_cursor(destination.name) or {}
        cursor_ts = existing.get("last_pushed_created_at")
        cursor_pid = existing.get("last_pushed_pulse_id")
    result.cursor_start = cursor_ts

    # Health-check the destination up front so we fail fast.
    ok, msg = destination.healthcheck()
    if not ok:
        result.error = f"healthcheck failed: {msg}"
        result.elapsed_sec = time.time() - t0
        if cursor_ts:
            store.set_ingestion_cursor(
                destination.name, cursor_ts, result.total_written,
                last_pushed_pulse_id=cursor_pid, last_error=result.error,
            )
        return result

    try:
        for _ in range(max_batches):
            rows = store.pulses_since(
                since_created_at=cursor_ts,
                since_pulse_id=cursor_pid,
                algo_id=algo_id,
                limit=batch_size,
            )
            if not rows:
                break
            result.total_read += len(rows)

            converted = [to_market_pulse_row(r) for r in rows]
            written = destination.write(converted)
            result.total_written += written
            result.batches += 1

            for r in rows:
                k = r.get("algo_id") or "unknown"
                result.per_algo[k] = result.per_algo.get(k, 0) + 1

            # Advance cursor to the last (created_at, pulse_id) in the batch.
            last = rows[-1]
            new_ts = last.get("created_at")
            new_pid = last.get("pulse_id")
            if new_ts and (new_ts, new_pid) != (cursor_ts, cursor_pid):
                cursor_ts, cursor_pid = new_ts, new_pid
            else:
                log.warning("no forward progress on cursor; aborting loop")
                break

            if progress_cb:
                progress_cb(result)
        result.cursor_end = cursor_ts
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"
        log.exception("push failed")
    finally:
        result.elapsed_sec = time.time() - t0
        if cursor_ts:
            store.set_ingestion_cursor(
                destination.name, cursor_ts, result.total_written,
                last_pushed_pulse_id=cursor_pid, last_error=result.error,
            )

    return result
