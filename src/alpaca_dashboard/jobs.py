"""In-process background-job registry for the /admin page.

Streamlit reruns scripts on every interaction, so we keep the job registry
module-global (``_REGISTRY``). Threads started here survive across reruns
for the life of the Streamlit server process.

Each job runs one algo's backtest. The UI polls ``snapshot()`` to render
progress, and calls ``request_stop(job_id)`` to abort.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import store
from .backtest import BacktestParams, run_single_algo


@dataclass
class JobHandle:
    job_id: str
    algo_id: str
    params: dict
    thread: threading.Thread
    stop_event: threading.Event = field(default_factory=threading.Event)
    progress: dict[str, Any] = field(default_factory=lambda: {"done": 0, "target": 0, "msg": ""})
    status: str = "running"          # running | done | error | stopped
    summary: dict | None = None
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str | None = None


_REGISTRY: dict[str, JobHandle] = {}
_LOCK = threading.Lock()


def start_algo_job(
    algo_id: str,
    tickers: list[str],
    days: int,
    target_pulses: int,
) -> JobHandle:
    job_id = f"job_{algo_id}_{uuid.uuid4().hex[:6]}"
    params = dict(tickers=tickers, days=days, target_pulses=target_pulses)
    store.create_job(job_id, algo_id, params)

    handle = JobHandle(job_id=job_id, algo_id=algo_id, params=params,
                       thread=None)  # type: ignore[arg-type]

    def _progress(done: int, target: int, msg: str) -> None:
        handle.progress = {"done": done, "target": target, "msg": msg}

    def _stop() -> bool:
        return handle.stop_event.is_set()

    def _run() -> None:
        try:
            p = BacktestParams(
                algo_id=algo_id,
                tickers=tickers,
                days=days,
                target_pulses=target_pulses,
                job_id=job_id,
                progress_cb=_progress,
                stop_cb=_stop,
            )
            summary = run_single_algo(p)
            handle.summary = summary
            handle.status = "stopped" if summary.get("aborted") else "done"
        except Exception as e:  # noqa: BLE001
            handle.status = "error"
            handle.error = f"{type(e).__name__}: {e}"
        finally:
            handle.ended_at = datetime.utcnow().isoformat()
            store.finish_job(job_id, handle.summary, handle.error)

    t = threading.Thread(target=_run, name=job_id, daemon=True)
    handle.thread = t
    with _LOCK:
        _REGISTRY[job_id] = handle
    t.start()
    return handle


def request_stop(job_id: str) -> bool:
    with _LOCK:
        h = _REGISTRY.get(job_id)
    if not h:
        return False
    h.stop_event.set()
    return True


def snapshot() -> list[dict]:
    """Serialisable view of live + recent jobs."""
    with _LOCK:
        handles = list(_REGISTRY.values())
    out = []
    for h in sorted(handles, key=lambda x: x.started_at, reverse=True):
        out.append({
            "job_id": h.job_id,
            "algo_id": h.algo_id,
            "status": h.status,
            "started_at": h.started_at,
            "ended_at": h.ended_at,
            "progress": h.progress,
            "summary": h.summary,
            "error": h.error,
            "params": h.params,
            "alive": h.thread.is_alive() if h.thread else False,
        })
    return out


def get(job_id: str) -> dict | None:
    with _LOCK:
        h = _REGISTRY.get(job_id)
    if not h:
        return None
    return {
        "job_id": h.job_id,
        "algo_id": h.algo_id,
        "status": h.status,
        "progress": h.progress,
        "summary": h.summary,
        "error": h.error,
        "alive": h.thread.is_alive() if h.thread else False,
    }
