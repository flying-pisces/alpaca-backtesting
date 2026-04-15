"""TTL cache around ``historical_data.fetch_history``.

Five algos scanning the same 30-ticker universe every cycle would otherwise
mean 150 API hits per cycle. With a 5-minute TTL one ticker's bars are
fetched once; subsequent algos read from the in-memory cache. This is the
rate-limit layer for the live engine.

Thread-safe (the live engine runs in a daemon thread, Streamlit may also
hit the cache from the request thread).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .historical_data import fetch_history as _raw_fetch


class TTLCache:
    def __init__(self, ttl_sec: float = 300.0):
        self.ttl = ttl_sec
        self._lock = threading.Lock()
        self._data: dict[tuple, tuple[float, object]] = {}
        self._hits = 0
        self._misses = 0

    def get_or_fetch(self, symbol: str, days: int, fetcher=_raw_fetch) -> Optional[dict]:
        key = (symbol.upper(), int(days))
        now = time.time()
        with self._lock:
            hit = self._data.get(key)
            if hit and (now - hit[0]) < self.ttl:
                self._hits += 1
                return hit[1]  # type: ignore[return-value]
        # Fetch outside the lock so concurrent callers for different symbols
        # don't serialise on the network call.
        value = fetcher(symbol, days)
        with self._lock:
            self._data[key] = (time.time(), value)
            self._misses += 1
        return value

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._data),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "ttl_sec": self.ttl,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Module-level singleton — one cache per process.
BARS_CACHE = TTLCache(ttl_sec=300.0)


def fetch_bars(symbol: str, days: int = 180) -> Optional[dict]:
    """Public entrypoint used by the live engine. Caches by ``(symbol, days)``."""
    return BARS_CACHE.get_or_fetch(symbol, days)
