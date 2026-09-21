from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from uuid import uuid4
from pathlib import Path
from typing import Any, Protocol

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
        path.touch(exist_ok=True)

    def publish(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class EventDelivery:
    """Persistent at-least-once delivery with idempotent enqueue."""

    flush_chunk = 200

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
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(outbox)")}
        additions = {
            "attempts": "INTEGER NOT NULL DEFAULT 0",
            "next_attempt_at": "REAL NOT NULL DEFAULT 0",
            "last_error": "TEXT",
        }
        missing = additions.keys() - columns
        if missing:
            # Back up existing events before an additive, transactional migration.
            if self.connection.execute("SELECT 1 FROM outbox LIMIT 1").fetchone():
                backup_path = database.with_name(f"{database.name}.backup-{uuid4().hex}.sqlite3")
                with sqlite3.connect(backup_path) as backup:
                    self.connection.backup(backup)
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                for name in sorted(missing):
                    self.connection.execute(f"ALTER TABLE outbox ADD COLUMN {name} {additions[name]}")
        self.connection.execute("PRAGMA journal_mode=WAL")

    def enqueue(self, records: list[EventRecord]) -> int:
        """Persist locally without invoking the destination; return newly queued rows."""
        before = self.connection.total_changes
        with self.connection:
            for record in records:
                self.connection.execute(
                    "INSERT OR IGNORE INTO outbox(idempotency_key, payload) VALUES (?, ?)",
                    (record.idempotency_key, json.dumps(record.to_dict(), ensure_ascii=False)),
                )
        return self.connection.total_changes - before

    def submit(self, records: list[EventRecord]) -> int:
        self.enqueue(records)
        return self.flush()

    def flush(self) -> int:
        delivered = 0
        while True:
            rows = self.connection.execute(
                "SELECT idempotency_key, payload FROM outbox WHERE delivered = 0 ORDER BY rowid LIMIT ?",
                (self.flush_chunk,),
            ).fetchall()
            if not rows:
                break
            for key, payload in rows:
                self.destination.publish(json.loads(payload))
                with self.connection:
                    self.connection.execute(
                        "UPDATE outbox SET delivered = 1 WHERE idempotency_key = ?", (key,)
                    )
                delivered += 1
        return delivered

    def drain_due(self, *, now: float | None = None, retry_base: float = 1.0,
                  retry_max: float = 60.0, should_stop: Callable[[], bool] = lambda: False) -> int:
        """Attempt a bounded FIFO batch. A failed head cannot be overtaken.

        A single background consumer owns this method. Destination failures are
        persisted; SQLite failures propagate so storage faults cannot look healthy.
        """
        delivered = 0
        for _ in range(self.flush_chunk):
            if should_stop():
                break
            row = self.connection.execute(
                "SELECT idempotency_key, payload, attempts, next_attempt_at "
                "FROM outbox WHERE delivered = 0 ORDER BY rowid LIMIT 1"
            ).fetchone()
            current = time.time() if now is None else now
            if row is None or row[3] > current:
                break
            key, payload, attempts, _ = row
            record = json.loads(payload)
            try:
                self.destination.publish(record)
            except Exception as error:
                delay = min(retry_max, retry_base * (2 ** min(attempts, 20)))
                failed_at = time.time() if now is None else now
                with self.connection:
                    self.connection.execute(
                        "UPDATE outbox SET attempts=attempts+1, next_attempt_at=?, last_error=? "
                        "WHERE idempotency_key=?",
                        (failed_at + delay, type(error).__name__, key),
                    )
                break
            with self.connection:
                self.connection.execute(
                    "UPDATE outbox SET delivered=1, attempts=attempts+1, "
                    "next_attempt_at=0, last_error=NULL WHERE idempotency_key=?", (key,),
                )
            delivered += 1
        return delivered

    def retry_status(self) -> dict:
        row = self.connection.execute(
            "SELECT idempotency_key, attempts, next_attempt_at, last_error "
            "FROM outbox WHERE delivered=0 ORDER BY rowid LIMIT 1"
        ).fetchone()
        return {"pending": self.pending_count(), "head": None if row is None else {
            "idempotencyKey": row[0], "attempts": row[1],
            "nextAttemptAt": row[2], "errorType": row[3],
        }}

    def pending_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM outbox WHERE delivered = 0").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EventDelivery:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                # Drain whatever the caller produced before closing cleanly.
                self.flush()
        finally:
            self.close()
