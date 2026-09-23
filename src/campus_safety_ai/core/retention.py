"""Conservative, recoverable retention for explicitly registered evidence only.

No source discovery, outbox pruning, log truncation, or permanent deletion occurs.
Stop video producers before applying maintenance. The sender lease and a SQLite
write transaction prevent delivery/enqueue from racing the maintenance snapshot.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import stat
import time
from collections import defaultdict
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.event_delivery import read_outbox_status
from campus_safety_ai.core.outbox_lease import OutboxLease

REGISTRY_NAME = ".evidence-ownership.jsonl"


def _root(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("managed paths must not contain symlinks or '..'")
    if not path.is_dir():
        raise ValueError("managed root must be an existing directory")
    return path


def _relative(root: Path, path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("evidence path contains '..'")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("path is outside the managed directory") from None
    if not relative.parts:
        raise ValueError("a file path is required")
    return relative


@contextmanager
def _parent_fd(root: Path, relative: Path):
    """Pin every directory without following symlinks, including during rename."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _fingerprint(root: Path, relative: Path) -> dict:
    with _parent_fd(root, relative) as parent:
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("evidence must be a regular file with one hard link")
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("evidence changed during verification")
            return {"sha256": digest, "size": after.st_size, "inode": after.st_ino,
                    "device": after.st_dev, "mtimeNs": after.st_mtime_ns}


def _append_jsonl(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("journal must be a regular file")
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def register_managed_evidence(evidence_root: Path, path: Path) -> None:
    """Called by an evidence writer immediately after creating its own artifact.

    Never register input/source videos or bulk-adopt an existing directory. A
    receipt proves the exact bytes produced by the caller; old files without a
    receipt remain outside automatic retention.
    """
    root = _root(evidence_root)
    relative = _relative(root, path)
    if relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4"}:
        raise ValueError("unsupported generated evidence type")
    fingerprint = _fingerprint(root, relative)
    _append_jsonl(root / REGISTRY_NAME, {"path": str(relative), **fingerprint, "registeredAt": time.time()})


def _registrations(root: Path) -> dict[str, dict]:
    path = root / REGISTRY_NAME
    if not path.exists() and not path.is_symlink():
        return {}
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    registrations = {}
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            relative = Path(record["path"])
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("invalid evidence ownership receipt path")
            if (not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64
                    or not isinstance(record.get("size"), int) or record["size"] < 0
                    or not isinstance(record.get("registeredAt"), (int, float))
                    or not math.isfinite(record["registeredAt"])):
                raise ValueError("invalid evidence ownership receipt")
            registrations[str(relative)] = record
    return registrations


def _plan(connection: sqlite3.Connection, runtime: Path, evidence: Path,
          now: float, retention_seconds: float) -> dict:
    status = read_outbox_status(connection, now=now)
    result = {"mode": "dry-run", "outbox": status, "candidates": [], "skipped": [],
              "quarantined": [], "blockedReasons": [], "candidateBytes": 0,
              "outboxRowsDeleted": 0, "logsDeleted": 0, "bytesFreed": 0}
    logs = ("events.jsonl", "observations.jsonl", "delivered-events.jsonl", "unsubmitted-events.jsonl")
    result["logBytes"] = {name: (runtime / name).lstat().st_size for name in logs
                          if (runtime / name).is_file() and not (runtime / name).is_symlink()}
    unsubmitted = runtime / "unsubmitted-events.jsonl"
    if unsubmitted.is_symlink() or (unsubmitted.exists() and unsubmitted.stat().st_size):
        result["blockedReasons"].append("unsubmitted-events.jsonl must be reconciled before evidence retention")
        return result
    columns = {row[1] for row in connection.execute("PRAGMA table_info(outbox)")}
    if not {"event_id", "delivered_at"} <= columns:
        result["blockedReasons"].append("legacy outbox has no delivery timestamp; no evidence can be expired")
        return result
    registrations = _registrations(evidence)
    events = {}
    references = defaultdict(set)
    for key, payload, delivered, delivered_at, event_id in connection.execute(
            "SELECT idempotency_key, payload, delivered, delivered_at, event_id FROM outbox ORDER BY rowid"):
        value = json.loads(payload)
        if (not isinstance(value, dict) or not isinstance(value.get("evidenceUris", []), list)
                or any(not isinstance(uri, str) for uri in value.get("evidenceUris", []))):
            raise ValueError("malformed event evidence references; retention refused")
        record = EventRecord.from_dict(value)
        if record.event_id != event_id or record.idempotency_key != key:
            raise ValueError("outbox payload identity mismatch; retention refused")
        event = events.setdefault(event_id, {"last": None, "safe": True, "deliveredAt": 0})
        previous = event["last"]
        if (previous is None and (record.phase != "START" or record.revision != 1)) or (
                previous is not None and (record.revision != previous.revision + 1 or previous.phase == "END")):
            event["safe"] = False
        if delivered != 1 or delivered_at is None or not math.isfinite(delivered_at):
            event["safe"] = False
        else:
            event["deliveredAt"] = max(event["deliveredAt"], delivered_at)
        event["last"] = record
        for uri in record.evidence_uris:
            if "://" in uri:
                result["skipped"].append({"path": uri, "reason": "remote URI is unmanaged"})
                continue
            path = Path(uri)
            try:
                relative = _relative(evidence, path)
            except ValueError:
                result["skipped"].append({"path": uri, "reason": "outside managed evidence directory"})
                continue
            references[str(relative)].add(event_id)
    cutoff = now - retention_seconds
    for relative, event_ids in references.items():
        item = {"path": str(evidence / relative), "eventIds": sorted(event_ids)}
        receipt = registrations.get(relative)
        reason = None
        if receipt is None:
            reason = "no generated evidence ownership receipt"
        else:
            for event_id in event_ids:
                event = events[event_id]
                last = event["last"]
                if not event["safe"] or last.phase != "END" or last.status != "CLOSED":
                    reason = "associated event is active, incomplete, pending, or has unknown delivery age"
                    break
                if max(last.ended_at.timestamp(), event["deliveredAt"], receipt["registeredAt"]) >= cutoff:
                    reason = "retention interval has not elapsed since end, delivery, and evidence creation"
                    break
        if reason is None:
            try:
                actual = _fingerprint(evidence, Path(relative))
                if any(actual.get(field) != receipt.get(field) for field in actual):
                    reason = "evidence changed since its generation receipt"
            except (OSError, ValueError):
                reason = "missing, symlinked, multiply linked, or unreadable evidence"
        if reason:
            result["skipped"].append({**item, "reason": reason})
        else:
            result["candidates"].append({**item, "relativePath": relative, **actual,
                                         "reason": "all referring events ended, delivered, and expired"})
            result["candidateBytes"] += actual["size"]
    return result


def maintain_runtime(*, runtime_root: Path, database: Path, evidence_root: Path,
                     retention_seconds: float, apply: bool = False, now: float | None = None) -> dict:
    """Plan retention, or move the exact revalidated files to a recovery directory.

    `runtime_root` must be the dedicated output directory for this one outbox.
    Nothing recursively discovers other runtimes, evidence, logs, or datasets.
    """
    timestamp = time.time() if now is None else now
    if not math.isfinite(retention_seconds) or retention_seconds <= 0 or not math.isfinite(timestamp):
        raise ValueError("retention_seconds must be finite and positive, and now must be finite")
    runtime, evidence = _root(runtime_root), _root(evidence_root)
    _relative(runtime, evidence)
    database = database.absolute()
    relative_database = _relative(runtime, database)
    # Inspect via the same no-symlink rule as artifacts, without hashing a live DB.
    with _parent_fd(runtime, relative_database) as parent:
        metadata = os.stat(relative_database.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("outbox must be a regular, unlinked file in runtime_root")
    lease = OutboxLease(database) if apply else None
    try:
        if lease is not None:
            lease.acquire()
        mode = "rw" if apply else "ro"
        with closing(sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True)) as connection:
            connection.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
            try:
                report = _plan(connection, runtime, evidence, timestamp, retention_seconds)
                if not apply:
                    return report
                report["mode"] = "apply"
                if not report["candidates"]:
                    return report
                quarantine_root = runtime / "quarantine"
                if quarantine_root.is_symlink():
                    raise ValueError("quarantine directory must not be a symlink")
                quarantine_root.mkdir(exist_ok=True)
                quarantine = quarantine_root / f"evidence-{uuid4().hex}"
                quarantine.mkdir(mode=0o700)
                report["quarantineDirectory"] = str(quarantine)
                receipt_path = quarantine / "receipt.jsonl"
                for index, item in enumerate(report["candidates"]):
                    relative = Path(item["relativePath"])
                    actual = _fingerprint(evidence, relative)
                    if any(actual[field] != item[field] for field in actual):
                        raise ValueError("evidence changed after planning; remaining moves refused")
                    target = quarantine / f"{index:06d}-{relative.name}"
                    receipt = {"source": item["path"], "target": str(target),
                               "eventIds": item["eventIds"], **actual}
                    # A durable intent permits recovery even if interrupted after rename.
                    _append_jsonl(receipt_path, {"state": "planned", **receipt})
                    with _parent_fd(evidence, relative) as parent, _parent_fd(quarantine, Path(target.name)) as dest:
                        before = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
                        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (
                                actual["inode"], actual["device"], actual["size"], actual["mtimeNs"]):
                            raise ValueError("evidence changed immediately before quarantine")
                        os.rename(relative.name, target.name, src_dir_fd=parent, dst_dir_fd=dest)
                        os.fsync(parent)
                        os.fsync(dest)
                    _append_jsonl(receipt_path, {"state": "moved", **receipt})
                    report["quarantined"].append(receipt)
                return report
            finally:
                connection.rollback()  # Outbox rows are never updated or pruned.
    finally:
        if lease is not None:
            lease.close()
