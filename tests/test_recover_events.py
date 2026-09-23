import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import test_delivery

from campus_safety_ai.apps import recover_events as recovery
from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination, OutboxCapacityError
from campus_safety_ai.core.outbox_lease import OutboxLease


def fixture(root: Path, *, existing_start=True):
    root.mkdir()
    start = test_delivery.DeliveryTests().record()
    end = replace(start, revision=2, phase="END", status="CLOSED",
                  ended_at=start.observed_at, idempotency_key="event-1:2")
    sender = EventDelivery(root / "outbox.sqlite3", InMemoryDestination())
    try:
        if existing_start:
            sender.enqueue([start])
    finally:
        sender.close()
    audit = [{**record.to_dict(), "sourceEpoch": 1, "reason": "fixture"} for record in (start, end)]
    (root / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audit))
    records = [end] if existing_start else [start, end]
    (root / "unsubmitted-events.jsonl").write_text("".join(json.dumps(row.to_dict()) + "\n" for row in records))
    return start, end


def rows(root):
    with closing(sqlite3.connect(root / "outbox.sqlite3")) as connection:
        return connection.execute("SELECT idempotency_key, payload, delivered FROM outbox ORDER BY rowid").fetchall()


def test_dry_run_validates_end_only_without_writes_or_delivery(tmp_path):
    root = tmp_path / "run"
    fixture(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with patch.object(InMemoryDestination, "publish", side_effect=AssertionError("must not send")):
        report = recovery.recover_events(root)
    assert report["mode"] == "dry-run" and report["status"] == "ready"
    assert report["new_records"] == 1 and report["pending_records"] == 1
    assert report["network_sends"] == 0
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_apply_restores_only_original_end_keeps_journal_and_retries_idempotently(tmp_path):
    root = tmp_path / "run"
    start, end = fixture(root)
    journal = (root / "unsubmitted-events.jsonl").read_bytes()
    with patch.object(InMemoryDestination, "publish", side_effect=AssertionError("must not send")):
        report = recovery.recover_events(root, apply=True)
        assert report["inserted_records"] == 1
        # Even a subsequently lower cap cannot reject already persisted duplicates.
        again = recovery.recover_events(root, apply=True, max_pending_records=1)
        assert again["inserted_records"] == 0 and again["capacity_allowed"]
    stored = rows(root)
    assert [row[0] for row in stored] == [start.idempotency_key, end.idempotency_key]
    assert [row[2] for row in stored] == [0, 0]
    assert [json.loads(row[1]) for row in stored] == [start.to_dict(), end.to_dict()]
    assert (root / "unsubmitted-events.jsonl").read_bytes() == journal
    receipts = [json.loads(line) for line in (root / recovery.RECEIPT_NAME).read_text().splitlines()]
    assert [row["state"] for row in receipts] == ["planned", "enqueued", "planned", "enqueued"]
    assert all(row["source"]["sha256"] == report["source"]["sha256"] for row in receipts)


def test_full_queue_rejects_whole_batch_and_sender_lease_is_exclusive(tmp_path):
    root = tmp_path / "run"
    fixture(root)
    before = rows(root)
    report = recovery.recover_events(root, max_pending_records=1)
    assert report["status"] == "blocked" and not report["capacity_allowed"]
    with pytest.raises(OutboxCapacityError):
        recovery.recover_events(root, apply=True, max_pending_records=1)
    assert rows(root) == before
    lease = OutboxLease(root / "outbox.sqlite3").acquire()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            recovery.recover_events(root, apply=True)
    finally:
        lease.close()
    assert rows(root) == before


def test_mismatched_audit_duplicate_payload_and_outbox_payload_are_refused(tmp_path):
    for mode in ("audit", "duplicate", "outbox"):
        root = tmp_path / mode
        start, end = fixture(root, existing_start=False)
        altered = replace(start, confidence=.1)
        if mode == "audit":
            (root / "unsubmitted-events.jsonl").write_text(json.dumps(altered.to_dict()) + "\n")
        elif mode == "duplicate":
            with (root / "unsubmitted-events.jsonl").open("a") as handle:
                handle.write(json.dumps(altered.to_dict()) + "\n")
        else:
            sender = EventDelivery(root / "outbox.sqlite3", InMemoryDestination())
            sender.enqueue([altered])
            sender.close()
        before = rows(root)
        with pytest.raises(ValueError, match="match|conflicting"):
            recovery.recover_events(root, apply=True)
        assert rows(root) == before


def test_missing_start_and_reversed_generated_order_are_refused(tmp_path):
    for mode in ("missing", "reversed"):
        root = tmp_path / mode
        start, end = fixture(root, existing_start=False)
        records = [end] if mode == "missing" else [end, start]
        (root / "unsubmitted-events.jsonl").write_text("".join(json.dumps(row.to_dict()) + "\n" for row in records))
        with pytest.raises(ValueError, match="START|generated order"):
            recovery.recover_events(root, apply=True)
        assert rows(root) == []


def test_commit_then_receipt_failure_can_be_retried_without_duplicate_events(tmp_path):
    root = tmp_path / "run"
    fixture(root, existing_start=False)
    original_receipt = recovery._receipt
    calls = 0

    def interrupted_receipt(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption after commit")
        return original_receipt(*args)

    with patch.object(recovery, "_receipt", interrupted_receipt), pytest.raises(OSError, match="after commit"):
        recovery.recover_events(root, apply=True)
    assert len(rows(root)) == 2
    assert (root / "unsubmitted-events.jsonl").is_file()
    report = recovery.recover_events(root, apply=True)
    assert report["inserted_records"] == 0
    assert len(rows(root)) == 2


def test_symlink_inputs_directory_and_size_cap_are_refused_without_enqueue(tmp_path):
    root = tmp_path / "run"
    fixture(root)
    with pytest.raises(ValueError, match="max_file_bytes"):
        recovery.recover_events(root, max_file_bytes=1)
    journal = root / "unsubmitted-events.jsonl"
    original = tmp_path / "source.jsonl"
    journal.rename(original)
    journal.symlink_to(original)
    with pytest.raises(ValueError, match="regular file"):
        recovery.recover_events(root, apply=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        recovery.recover_events(alias, apply=True)
    assert len(rows(root)) == 1
    assert original.is_file()


def test_atomic_enqueue_rechecks_payload_after_recovery_preflight(tmp_path):
    root = tmp_path / "run"
    start, end = fixture(root, existing_start=False)
    validate = recovery._validate

    def competing_producer(*args):
        result = validate(*args)
        sender = EventDelivery(root / "outbox.sqlite3", InMemoryDestination())
        sender.enqueue([replace(start, confidence=.1)])
        sender.close()
        return result

    with patch.object(recovery, "_validate", competing_producer), pytest.raises(ValueError, match="conflicting"):
        recovery.recover_events(root, apply=True)
    stored = rows(root)
    assert len(stored) == 1  # Rejected END was not partly inserted.
    assert json.loads(stored[0][1])["confidence"] == .1


def test_dry_run_includes_uncheckpointed_wal_without_modifying_source_files(tmp_path):
    root = tmp_path / "run"
    start, end = fixture(root, existing_start=False)
    sender = EventDelivery(root / "outbox.sqlite3", InMemoryDestination())
    try:
        sender.enqueue([start])
        assert (root / "outbox.sqlite3-wal").stat().st_size > 0
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        report = recovery.recover_events(root)
        assert report["already_present"] == 1
        assert report["new_records"] == 1
        assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    finally:
        sender.close()


def test_atomic_enqueue_refuses_competing_event_revision_after_preflight(tmp_path):
    root = tmp_path / "run"
    start, end = fixture(root)
    validate = recovery._validate

    def competing_producer(*args):
        result = validate(*args)
        sender = EventDelivery(root / "outbox.sqlite3", InMemoryDestination())
        sender.enqueue([replace(start, revision=2, phase="UPDATE", idempotency_key="competing:2")])
        sender.close()
        return result

    with patch.object(recovery, "_validate", competing_producer), pytest.raises(ValueError, match="reorder"):
        recovery.recover_events(root, apply=True)
    assert [row[0] for row in rows(root)] == ["event-1:1", "competing:2"]
