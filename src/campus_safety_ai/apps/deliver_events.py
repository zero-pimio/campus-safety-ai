"""Inspect an outbox or resume delivery without rerunning video inference."""
from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import threading
from pathlib import Path

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.settings import load_platform_settings


def inspect_outbox(database: Path) -> dict:
    # Read-only mode must not create or migrate a database just to inspect it.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(outbox)")}
        pending = connection.execute("SELECT COUNT(*) FROM outbox WHERE delivered=0").fetchone()[0]
        if "attempts" not in columns:
            return {"pending": pending, "schema": "legacy"}
        row = connection.execute(
            "SELECT attempts,next_attempt_at,last_error FROM outbox "
            "WHERE delivered=0 ORDER BY rowid LIMIT 1"
        ).fetchone()
        return {"pending": pending, "head": None if row is None else {
            "attempts": row[0], "nextAttemptAt": row[1], "errorType": row[2],
        }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-config", required=True, type=Path)
    parser.add_argument("--outbox", type=Path)
    parser.add_argument("--status", action="store_true", help="read only; never send events")
    parser.add_argument("--retry-base", type=float, default=1)
    parser.add_argument("--retry-max", type=float, default=60)
    args = parser.parse_args()
    settings = load_platform_settings(args.platform_config)
    database = args.outbox or settings.outbox
    if args.status:
        print(json.dumps(inspect_outbox(database), ensure_ascii=False))
        return
    if not database.is_file():
        parser.error("outbox does not exist; nothing to resume")
    stop = threading.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, lambda *_: stop.set())
    try:
        with BackgroundDelivery(database, build_platform_destination(settings),
                                retry_base=args.retry_base, retry_max=args.retry_max) as delivery:
            while not stop.wait(0.1):
                if delivery.pending_count() == 0:
                    break
            print(json.dumps(delivery.retry_status(), ensure_ascii=False))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
