"""Validate rejected events and, with --apply, enqueue them locally without sending."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import time
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination
from campus_safety_ai.core.outbox_lease import OutboxLease

DEFAULT_FILE_LIMIT = 64 * 1024 * 1024
RECEIPT_NAME = "recovery-receipts.jsonl"


def _root_descriptor(path: Path) -> int:
    if ".." in path.parts:
        raise ValueError("runtime directory must not contain '..'")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _regular(root_fd: int, name: str, *, optional: bool = False):
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        if optional:
            return None
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be a regular file with no symlink or extra hard link")
    return metadata


def _read_jsonl(root_fd: int, name: str, limit: int) -> tuple[list[dict], dict]:
    _regular(root_fd, name)
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    digest, values, size = hashlib.sha256(), [], 0
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > limit:
            raise ValueError(f"{name} exceeds max_file_bytes")
        while line := handle.readline(limit - size + 1):
            size += len(line)
            if size > limit:
                raise ValueError(f"{name} grew beyond max_file_bytes")
            digest.update(line)
            if not line.strip():
                raise ValueError(f"{name} contains a blank record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{name} contains a non-object record")
            values.append(value)
        after = os.fstat(handle.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"{name} changed during inspection; stop its producer before recovery")
    return values, {"name": name, "sha256": digest.hexdigest(), "bytes": size, "lines": len(values)}


def _canonical(value: dict, *, audit: bool = False) -> tuple[EventRecord, str]:
    payload = {key: item for key, item in value.items() if not audit or key not in {"sourceEpoch", "reason"}}
    try:
        record = EventRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid EventRecord in recovery inputs") from error
    expected = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if expected != json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")):
        raise ValueError("recovery payload differs from the complete generated EventRecord schema")
    return record, expected


@contextmanager
def _database_snapshot(root_fd: int, limit: int):
    # SQLite mode=ro can still create WAL/SHM sidecars beside the source. Read
    # bounded raw files into a private temporary directory instead. Include WAL
    # so committed but uncheckpointed records are never silently omitted.
    names = ("outbox.sqlite3", "outbox.sqlite3-wal")
    before = {name: _regular(root_fd, name, optional=name.endswith("-wal")) for name in names}
    with tempfile.TemporaryDirectory(prefix="outbox-recovery-preview-") as temporary:
        for name, metadata in before.items():
            if metadata is None:
                continue
            if metadata.st_size > limit:
                raise ValueError(f"{name} exceeds max_file_bytes")
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            with os.fdopen(descriptor, "rb") as source, (Path(temporary) / name).open("wb") as target:
                remaining = limit
                while block := source.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError(f"{name} grew beyond max_file_bytes")
                    target.write(block)
        def identity(item):
            return None if item is None else (item.st_ino, item.st_size, item.st_mtime_ns)

        for name, old in before.items():
            new = _regular(root_fd, name, optional=name.endswith("-wal"))
            if identity(old) != identity(new):
                raise ValueError("outbox changed during snapshot; stop producers before recovery")
        database = Path(temporary) / "outbox.sqlite3"
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            yield connection


def _validate(root_fd: int, database: Path, file_limit: int, max_records: int, max_bytes: int):
    rejected, source_identity = _read_jsonl(root_fd, "unsubmitted-events.jsonl", file_limit)
    audit, audit_identity = _read_jsonl(root_fd, "events.jsonl", file_limit)
    generated = {}
    for position, value in enumerate(audit):
        record, canonical = _canonical(value, audit=True)
        previous = generated.setdefault(record.idempotency_key, (canonical, position))
        if previous[0] != canonical:
            raise ValueError("generated audit has conflicting payloads for one idempotency key")
    records, canonical_by_key, last_position = [], {}, -1
    for value in rejected:
        record, canonical = _canonical(value)
        key = record.idempotency_key
        if key in canonical_by_key:
            if canonical_by_key[key] != canonical:
                raise ValueError("rejected journal has conflicting payloads for one idempotency key")
            continue
        matching = generated.get(key)
        if matching is None or matching[0] != canonical:
            raise ValueError("rejected event does not exactly match the generated events audit")
        if matching[1] < last_position:
            raise ValueError("rejected events are not in their generated order")
        last_position = matching[1]
        canonical_by_key[key] = canonical
        records.append(record)
    _regular(root_fd, "outbox.sqlite3")
    with _database_snapshot(root_fd, file_limit) as connection:
        connection.execute("BEGIN")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(outbox)")}
        if not {"idempotency_key", "payload", "delivered", "attempts", "next_attempt_at",
                "last_error", "event_id", "enqueued_at", "delivered_at"} <= columns:
            raise ValueError("outbox needs a separate schema migration before recovery")
        existing, histories = set(), {}
        for event_id in dict.fromkeys(record.event_id for record in records):
            history = []
            for key, payload in connection.execute(
                    "SELECT idempotency_key, payload FROM outbox WHERE event_id=? ORDER BY rowid", (event_id,)):
                stored, canonical = _canonical(json.loads(payload))
                if stored.idempotency_key != key or stored.event_id != event_id:
                    raise ValueError("outbox payload identity mismatch")
                if key in canonical_by_key:
                    if canonical != canonical_by_key[key]:
                        raise ValueError("existing outbox idempotency key has a conflicting payload")
                    existing.add(key)
                history.append(stored)
            histories[event_id] = history
        # Catch a conflicting idempotency key indexed under a different event.
        for record in records:
            if record.idempotency_key not in existing:
                found = connection.execute("SELECT payload FROM outbox WHERE idempotency_key=?",
                                           (record.idempotency_key,)).fetchone()
                if found is not None:
                    raise ValueError("existing outbox idempotency key belongs to another event")
        incoming = [record for record in records if record.idempotency_key not in existing]
        for record in incoming:
            histories[record.event_id].append(record)
        for history in histories.values():
            previous = None
            for record in history:
                if previous is None:
                    if record.revision != 1 or record.phase != "START":
                        raise ValueError("recovery would leave an event without its original START")
                elif record.revision != previous.revision + 1 or previous.phase == "END":
                    raise ValueError("recovery would reorder or skip an event revision")
                previous = record
        pending, pending_bytes = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(payload AS BLOB))), 0) FROM outbox WHERE delivered=0"
        ).fetchone()
    incoming_bytes = sum(len(json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8")) for record in incoming)
    capacity = not incoming or (pending + len(incoming) <= max_records and pending_bytes + incoming_bytes <= max_bytes)
    keys_hash = hashlib.sha256("\n".join(record.idempotency_key for record in records).encode()).hexdigest()
    return records, {"status": "ready" if capacity else "blocked", "mode": "dry-run",
                     "source": source_identity, "audit": audit_identity,
                     "unique_records": len(records), "already_present": len(existing),
                     "new_records": len(incoming), "new_payload_bytes": incoming_bytes,
                     "pending_records": pending, "pending_bytes": pending_bytes,
                     "limits": {"max_pending_records": max_records, "max_pending_bytes": max_bytes,
                                "max_file_bytes": file_limit}, "capacity_allowed": capacity,
                     "idempotency_keys_sha256": keys_hash, "source_retained": True, "network_sends": 0}


def _receipt(root_fd: int, value: dict):
    _regular(root_fd, RECEIPT_NAME, optional=True)
    descriptor = os.open(RECEIPT_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                         0o600, dir_fd=root_fd)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.fsync(root_fd)


def recover_events(runtime_dir: Path, *, apply: bool = False, max_file_bytes: int = DEFAULT_FILE_LIMIT,
                   max_pending_records: int = 10000, max_pending_bytes: int = 64 * 1024 * 1024) -> dict:
    for value in (max_file_bytes, max_pending_records, max_pending_bytes):
        if type(value) is not int or value < 1:
            raise ValueError("recovery limits must be positive integers")
    root = runtime_dir.absolute()
    descriptor = _root_descriptor(root)
    database = root / "outbox.sqlite3"
    lease = OutboxLease(database) if apply else None
    try:
        for name in ("outbox.sqlite3", "outbox.sqlite3-wal", "outbox.sqlite3-shm",
                     "outbox.sqlite3.worker.lock", RECEIPT_NAME):
            metadata = _regular(descriptor, name, optional=name != "outbox.sqlite3")
            if name == RECEIPT_NAME and metadata and metadata.st_size > max_file_bytes:
                raise ValueError("recovery receipt exceeds max_file_bytes")
        if lease is not None:
            lease.acquire()
        records, report = _validate(descriptor, database, max_file_bytes, max_pending_records, max_pending_bytes)
        report["runtime_dir"] = str(root)
        if not apply:
            return report
        attempt = uuid4().hex
        planned = {"attempt": attempt, "recorded_at": time.time(), "state": "planned", **report}
        _receipt(descriptor, planned)
        sender = EventDelivery(database, InMemoryDestination(), max_pending_records=max_pending_records,
                               max_pending_bytes=max_pending_bytes)
        try:
            # Do not enter its context manager: __exit__ would flush. Strict
            # payload checks run in the same transaction as capacity and insert.
            inserted = sender.enqueue(records, require_matching_payload=True, require_contiguous_revisions=True)
        finally:
            sender.close()
        report.update(status="enqueued", mode="apply", inserted_records=inserted,
                      receipt=str(root / RECEIPT_NAME), attempt=attempt)
        _receipt(descriptor, {"attempt": attempt, "recorded_at": time.time(), "state": "enqueued", **report})
        return report
    finally:
        if lease is not None:
            lease.close()
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_FILE_LIMIT)
    parser.add_argument("--max-pending-records", type=int, default=10000)
    parser.add_argument("--max-pending-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    try:
        report = recover_events(args.runtime_dir, apply=args.apply, max_file_bytes=args.max_file_bytes,
                                max_pending_records=args.max_pending_records,
                                max_pending_bytes=args.max_pending_bytes)
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        parser.exit(2, f"recovery refused: {type(error).__name__}: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
