from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing, contextmanager
from uuid import uuid4
from pathlib import Path
from typing import Any, Protocol

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.outbox_lease import OutboxLease


def _file_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # A concurrent WAL checkpoint/connection close can remove sidecars.
            pass
    return total


def read_outbox_status(connection: sqlite3.Connection, now: float | None = None) -> dict:
    """Read a consistent queue snapshot without migrating or writing the database."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(outbox)")}
    if "attempts" not in columns:
        pending = connection.execute("SELECT COUNT(*) FROM outbox WHERE delivered=0").fetchone()[0]
        return {"pending": pending, "schema": "legacy"}
    cutoff = time.time() if now is None else now
    if not math.isfinite(cutoff):
        raise ValueError("now must be finite")
    partition = "AND prior.event_id = current.event_id" if "event_id" in columns else ""
    queued_at = "enqueued_at" if "enqueued_at" in columns else "NULL"
    row = connection.execute(f"""
        WITH heads AS (
            SELECT current.next_attempt_at FROM outbox AS current
            WHERE current.delivered=0 AND NOT EXISTS (
                SELECT 1 FROM outbox AS prior
                WHERE prior.delivered=0 AND prior.rowid < current.rowid {partition}
            )
        ), totals AS (
            SELECT COUNT(CASE WHEN delivered=0 THEN 1 END) AS pending,
                   COALESCE(SUM(attempts), 0) AS attempts,
                   COALESCE(SUM(CASE WHEN delivered=0 THEN length(CAST(payload AS BLOB)) END), 0) AS bytes,
                   MIN(CASE WHEN delivered=0 THEN {queued_at} END) AS oldest,
                   COUNT(CASE WHEN delivered=0 AND {queued_at} IS NULL THEN 1 END) AS unknown_age,
                   COUNT(*) AS records,
                   COALESCE(SUM(length(CAST(payload AS BLOB))), 0) AS all_bytes FROM outbox
        ), scheduling AS (
            SELECT COUNT(*) AS count, COALESCE(SUM(next_attempt_at <= ?), 0) AS ready,
                   MIN(next_attempt_at) AS next FROM heads
        ), first AS (
            SELECT idempotency_key, attempts, next_attempt_at, last_error
            FROM outbox WHERE delivered=0 ORDER BY rowid LIMIT 1
        )
        SELECT totals.pending, totals.attempts, scheduling.count, scheduling.ready, scheduling.next,
               first.idempotency_key, first.attempts, first.next_attempt_at, first.last_error,
               totals.bytes, totals.oldest, totals.unknown_age, totals.records, totals.all_bytes
        FROM totals CROSS JOIN scheduling LEFT JOIN first ON 1=1
    """, (cutoff,)).fetchone()
    status = {
        "pending": row[0], "totalAttempts": row[1], "readyEvents": row[3],
        "deferredEvents": row[2] - row[3], "blockedRecords": row[0] - row[2],
        "nextAttemptAt": row[4],
        "head": None if row[5] is None else {
            "idempotencyKey": row[5], "attempts": row[6],
            "nextAttemptAt": row[7], "errorType": row[8],
        },
        "pendingBytes": row[9], "oldestPendingAt": row[10],
        "oldestPendingAgeSeconds": None if row[10] is None else max(0.0, cutoff - row[10]),
        "unknownPendingAgeRecords": row[11], "totalRecords": row[12], "payloadBytes": row[13],
    }
    # Payload limits do not bound SQLite indexes, delivered history, or the WAL.
    paths = [Path(item[2]) for item in connection.execute("PRAGMA database_list") if item[1] == "main" and item[2]]
    status["databaseBytes"] = _file_bytes(paths)
    status["databaseAuxiliaryBytes"] = _file_bytes(
        [Path(str(path) + suffix) for path in paths for suffix in ("-wal", "-shm")])
    if "event_id" not in columns:
        status["schema"] = "fifo"
    return status


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


class OutboxCapacityError(RuntimeError):
    """No new records in this batch were persisted; stop/retry the producer."""

    def __init__(self, *, pending_records: int, pending_bytes: int, incoming_records: int,
                 incoming_bytes: int, max_pending_records: int | None, max_pending_bytes: int | None):
        self.pending_records = pending_records
        self.pending_bytes = pending_bytes
        self.incoming_records = incoming_records
        self.incoming_bytes = incoming_bytes
        self.max_pending_records = max_pending_records
        self.max_pending_bytes = max_pending_bytes
        super().__init__(
            "outbox capacity exceeded; entire new batch remains unsubmitted: "
            f"pending={pending_records} records/{pending_bytes} bytes, "
            f"incoming={incoming_records} records/{incoming_bytes} bytes, "
            f"limits={max_pending_records} records/{max_pending_bytes} bytes; "
            "stop inference and retain/retry the rejected records"
        )


class EventDelivery:
    """Persistent at-least-once delivery with idempotent enqueue."""

    flush_chunk = 200

    def __init__(self, database: Path, destination: Destination, *,
                 max_pending_records: int | None = None, max_pending_bytes: int | None = None,
                 _lease: OutboxLease | None = None) -> None:
        for value in (max_pending_records, max_pending_bytes):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("outbox limits must be positive integers or None")
        self.max_pending_records = max_pending_records
        self.max_pending_bytes = max_pending_bytes
        database = database.resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self._lease = _lease
        self.connection = sqlite3.connect(database)
        self.destination = destination
        try:
            self._initialize(database)
        except BaseException:
            self.connection.close()
            raise

    def _initialize(self, database: Path) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error TEXT,
                event_id TEXT NOT NULL DEFAULT '',
                enqueued_at REAL,
                delivered_at REAL
            )
            """
        )
        self.connection.commit()
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(outbox)")}
        additions = {
            "attempts": "INTEGER NOT NULL DEFAULT 0",
            "next_attempt_at": "REAL NOT NULL DEFAULT 0",
            "last_error": "TEXT",
            "event_id": "TEXT NOT NULL DEFAULT ''",
            "enqueued_at": "REAL",
            "delivered_at": "REAL",
        }
        missing = additions.keys() - columns
        if missing:
            # Back up existing events before an additive, transactional migration.
            if self.connection.execute("SELECT 1 FROM outbox LIMIT 1").fetchone():
                backup_path = database.with_name(f"{database.name}.backup-{uuid4().hex}.sqlite3")
                with closing(sqlite3.connect(backup_path)) as backup:
                    self.connection.backup(backup)
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                # Another producer may have migrated while this connection backed up.
                columns = {row[1] for row in self.connection.execute("PRAGMA table_info(outbox)")}
                missing = additions.keys() - columns
                for name in sorted(missing):
                    self.connection.execute(f"ALTER TABLE outbox ADD COLUMN {name} {additions[name]}")
                if "event_id" in missing:
                    cursor = self.connection.execute("SELECT idempotency_key, payload FROM outbox")
                    while rows := cursor.fetchmany(self.flush_chunk):
                        for key, payload in rows:
                            value = json.loads(payload)
                            event_id = value.get("eventId") if isinstance(value, dict) else None
                            if not isinstance(event_id, str) or not event_id:
                                raise ValueError("legacy outbox payload has no valid eventId")
                            self.connection.execute("UPDATE outbox SET event_id=? WHERE idempotency_key=?",
                                                    (event_id, key))
        with self.connection:
            self.connection.execute("CREATE INDEX IF NOT EXISTS outbox_pending ON outbox(delivered)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS outbox_event_pending "
                                    "ON outbox(event_id) WHERE delivered=0")
        self.connection.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _sending(self):
        if self._lease is not None:
            if not self._lease.held or self._lease.database != self.database:
                raise RuntimeError("delivery requires an active lease for this outbox")
            yield
        else:
            lease = OutboxLease(self.database)
            lease.acquire()
            try:
                yield
            finally:
                lease.close()

    def enqueue(self, records: list[EventRecord], *, require_matching_payload: bool = False,
                require_contiguous_revisions: bool = False) -> int:
        """Atomically persist a batch, or reject it without dropping existing records.

        Existing idempotency keys (including delivered history) consume no capacity.
        The write transaction serializes the capacity check across producers.
        """
        if not records:
            return 0
        unique = {}
        for record in records:
            if require_matching_payload and record.idempotency_key in unique:
                if record.to_dict() != unique[record.idempotency_key].to_dict():
                    raise ValueError("duplicate idempotency key has conflicting payloads in this batch")
            unique.setdefault(record.idempotency_key, record)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            incoming = []
            for key, record in unique.items():
                existing = self.connection.execute("SELECT payload FROM outbox WHERE idempotency_key=?", (key,)).fetchone()
                if existing:
                    if require_matching_payload and json.dumps(json.loads(existing[0]), sort_keys=True) != json.dumps(
                            record.to_dict(), sort_keys=True):
                        raise ValueError("existing idempotency key has a conflicting payload")
                    continue
                incoming.append((key, json.dumps(record.to_dict(), ensure_ascii=False), record.event_id))
            if not incoming:
                return 0
            if require_contiguous_revisions:
                tips = {}
                for key, _, event_id in incoming:
                    if event_id not in tips:
                        row = self.connection.execute(
                            "SELECT payload FROM outbox WHERE event_id=? ORDER BY rowid DESC LIMIT 1",
                            (event_id,),
                        ).fetchone()
                        tips[event_id] = EventRecord.from_dict(json.loads(row[0])) if row else None
                    record, previous = unique[key], tips[event_id]
                    if previous is None:
                        if record.revision != 1 or record.phase != "START":
                            raise ValueError("enqueue would omit the original event START")
                    elif previous.phase == "END" or record.revision != previous.revision + 1:
                        raise ValueError("enqueue would reorder an event revision")
                    tips[event_id] = record
            pending, pending_bytes = self.connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(length(CAST(payload AS BLOB))), 0) "
                "FROM outbox WHERE delivered=0"
            ).fetchone()
            incoming_bytes = sum(len(item[1].encode("utf-8")) for item in incoming)
            if ((self.max_pending_records is not None and pending + len(incoming) > self.max_pending_records)
                    or (self.max_pending_bytes is not None and pending_bytes + incoming_bytes > self.max_pending_bytes)):
                raise OutboxCapacityError(
                    pending_records=pending, pending_bytes=pending_bytes, incoming_records=len(incoming),
                    incoming_bytes=incoming_bytes, max_pending_records=self.max_pending_records,
                    max_pending_bytes=self.max_pending_bytes,
                )
            queued_at = time.time()
            for key, payload, event_id in incoming:
                self.connection.execute(
                    "INSERT INTO outbox(idempotency_key, payload, event_id, enqueued_at) VALUES (?, ?, ?, ?)",
                    (key, payload, event_id, queued_at),
                )
        return len(incoming)

    def submit(self, records: list[EventRecord]) -> int:
        self.enqueue(records)
        return self.flush()

    def flush(self) -> int:
        """Explicit synchronous flush; ignores backoff and propagates destination errors."""
        with self._sending():
            return self._flush()

    def _flush(self) -> int:
        delivered = 0
        while True:
            rows = self.connection.execute(
                "SELECT idempotency_key, payload FROM outbox WHERE delivered = 0 ORDER BY rowid LIMIT ?",
                (self.flush_chunk,),
            ).fetchall()
            if not rows:
                break
            for key, payload in rows:
                try:
                    self.destination.publish(json.loads(payload))
                except Exception as error:
                    with self.connection:
                        self.connection.execute(
                            "UPDATE outbox SET attempts=attempts+1, last_error=? WHERE idempotency_key=?",
                            (type(error).__name__, key),
                        )
                    raise
                with self.connection:
                    self.connection.execute(
                        "UPDATE outbox SET delivered=1, attempts=attempts+1, next_attempt_at=0, "
                        "last_error=NULL, delivered_at=? WHERE idempotency_key=?", (time.time(), key)
                    )
                delivered += 1
        return delivered

    def drain_due(self, *, now: float | None = None, retry_base: float = 1.0,
                  retry_max: float = 60.0, should_stop: Callable[[], bool] = lambda: False) -> int:
        """Attempt a bounded batch, keeping enqueue order within each event.

        A single background consumer owns this method. Destination failures are
        persisted; SQLite failures propagate so storage faults cannot look healthy.
        """
        for value in (retry_base, retry_max):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("retry delays must be finite and positive")
        if retry_max < retry_base:
            raise ValueError("retry_max must be at least retry_base")
        cutoff = time.time() if now is None else now
        if not math.isfinite(cutoff):
            raise ValueError("now must be finite")
        with self._sending():
            return self._drain_due(cutoff, now, retry_base, retry_max, should_stop)

    def _drain_due(self, cutoff, now, retry_base, retry_max, should_stop) -> int:
        delivered = 0
        for _ in range(self.flush_chunk):
            if should_stop():
                break
            row = self.connection.execute(
                "SELECT current.idempotency_key, current.payload, current.attempts, current.event_id "
                "FROM outbox AS current WHERE current.delivered=0 AND current.next_attempt_at <= ? "
                "AND NOT EXISTS (SELECT 1 FROM outbox AS prior WHERE prior.delivered=0 "
                "AND prior.event_id=current.event_id AND prior.rowid < current.rowid) "
                "ORDER BY current.rowid LIMIT 1", (cutoff,),
            ).fetchone()
            if row is None:
                break
            key, payload, attempts, event_id = row
            try:
                record = json.loads(payload)
                if (not isinstance(record, dict) or record.get("eventId") != event_id
                        or record.get("idempotencyKey") != key):
                    raise ValueError("outbox payload identity does not match its stored identity")
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
                continue
            with self.connection:
                self.connection.execute(
                    "UPDATE outbox SET delivered=1, attempts=attempts+1, "
                    "next_attempt_at=0, last_error=NULL, delivered_at=? WHERE idempotency_key=?",
                    (time.time() if now is None else now, key),
                )
            delivered += 1
        return delivered

    def retry_status(self) -> dict:
        return {**read_outbox_status(self.connection), "maxPendingRecords": self.max_pending_records,
                "maxPendingBytes": self.max_pending_bytes}

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
