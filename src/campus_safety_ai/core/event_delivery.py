from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from campus_safety_ai.contracts import EventRecord


class Destination(Protocol):
    def publish(self, record: dict) -> None: ...


class InMemoryDestination:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def publish(self, record: dict) -> None:
        self.records.append(record)


class JsonlDestination:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class EventDelivery:
    """Persistent at-least-once delivery with idempotent enqueue."""

    def __init__(self, database: Path, destination: Destination) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.destination = destination
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.commit()

    def submit(self, records: list[EventRecord]) -> int:
        with self.connection:
            for record in records:
                self.connection.execute(
                    "INSERT OR IGNORE INTO outbox(idempotency_key, payload) VALUES (?, ?)",
                    (record.idempotency_key, json.dumps(record.to_dict(), ensure_ascii=False)),
                )
        return self.flush()

    def flush(self) -> int:
        delivered = 0
        rows = self.connection.execute(
            "SELECT idempotency_key, payload FROM outbox WHERE delivered = 0 ORDER BY rowid"
        ).fetchall()
        for key, payload in rows:
            self.destination.publish(json.loads(payload))
            with self.connection:
                self.connection.execute("UPDATE outbox SET delivered = 1 WHERE idempotency_key = ?", (key,))
            delivered += 1
        return delivered

    def pending_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM outbox WHERE delivered = 0").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()

