"""Inspect an outbox or resume delivery without rerunning video inference."""
from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import read_outbox_status
from campus_safety_ai.settings import load_platform_settings


def inspect_outbox(database: Path) -> dict:
    # Read-only mode must not create or migrate a database just to inspect it.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return read_outbox_status(connection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-config", type=Path)
    parser.add_argument("--outbox", type=Path)
    parser.add_argument("--status", action="store_true", help="read only; never send events")
    parser.add_argument("--retry-base", type=float, default=1)
    parser.add_argument("--retry-max", type=float, default=60)
    args = parser.parse_args()
    if args.status:
        database = args.outbox
        if database is None:
            if args.platform_config is None:
                parser.error("--status requires --outbox or --platform-config")
            database = load_platform_settings(args.platform_config).outbox
        print(json.dumps(inspect_outbox(database), ensure_ascii=False))
        return
    if args.platform_config is None:
        parser.error("--platform-config is required to resume delivery")
    settings = load_platform_settings(args.platform_config)
    database = args.outbox or settings.outbox
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
