#!/usr/bin/env python3
"""CLI: push new backtest pulses into market_pulse.

Designed for cron / overnight use. Safe to re-run: uses a persistent cursor,
writes are idempotent.

Examples:
    # push everything new into the local market_pulse sqlite DB
    ./scripts/push_to_market_pulse.py \
        --sqlite ~/projects/market_pulse/pulses.db

    # only a single algo (e.g. after tuning dividend_play)
    ./scripts/push_to_market_pulse.py \
        --sqlite ~/projects/market_pulse/pulses.db \
        --algo dividend_play

    # start fresh (ignore the stored cursor) — use for the initial backfill
    ./scripts/push_to_market_pulse.py --sqlite <path> --reset-cursor
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from alpaca_dashboard import store  # noqa: E402
from alpaca_dashboard.ingestion import HttpDestination, SqliteDestination, push  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sqlite", metavar="PATH",
                       help="Path to a market_pulse pulses.db")
    group.add_argument("--http-url", metavar="URL",
                       help="(Stub) base URL of a future market_pulse ingestion endpoint")
    ap.add_argument("--http-token", default=None,
                    help="Bearer token for --http-url (reads HTTP_INGEST_TOKEN env if unset)")
    ap.add_argument("--algo", default=None,
                    help="Restrict push to a single algo_id")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--reset-cursor", action="store_true",
                    help="Ignore stored cursor and push from the beginning")
    ap.add_argument("--since", default=None,
                    help="Override cursor to this ISO timestamp (e.g. 2026-04-01T00:00:00)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be pushed without writing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.sqlite:
        dest = SqliteDestination(args.sqlite)
    else:
        token = args.http_token or __import__("os").getenv("HTTP_INGEST_TOKEN", "")
        if not token:
            print("--http-url requires --http-token or HTTP_INGEST_TOKEN env var", file=sys.stderr)
            return 2
        dest = HttpDestination(args.http_url, token)

    store.init_db()

    if args.dry_run:
        # Peek at rows that WOULD be pushed without writing.
        existing = store.get_ingestion_cursor(dest.name) or {}
        cursor_ts = None if args.reset_cursor else (args.since or existing.get("last_pushed_created_at"))
        cursor_pid = None if (args.reset_cursor or args.since) else existing.get("last_pushed_pulse_id")
        candidates = store.pulses_since(
            since_created_at=cursor_ts,
            since_pulse_id=cursor_pid,
            algo_id=args.algo,
            limit=args.batch_size,
        )
        by_algo: dict[str, int] = {}
        for r in candidates:
            by_algo[r["algo_id"]] = by_algo.get(r["algo_id"], 0) + 1
        print(json.dumps({
            "destination": dest.name,
            "dry_run": True,
            "cursor_from": cursor_ts,
            "candidates_in_first_batch": len(candidates),
            "per_algo": by_algo,
        }, indent=2))
        return 0

    result = push(
        dest,
        since=args.since,
        algo_id=args.algo,
        batch_size=args.batch_size,
        reset_cursor=args.reset_cursor,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
