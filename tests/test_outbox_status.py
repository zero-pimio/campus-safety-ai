import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campus_safety_ai.apps.deliver_events import inspect_outbox, main


class OutboxStatusTests(unittest.TestCase):
    def create_outbox(self, database: Path, *, legacy: bool = False, fifo: bool = False) -> None:
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute(
                "CREATE TABLE outbox (idempotency_key TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)"
            )
            if legacy:
                connection.execute("INSERT INTO outbox VALUES ('a:1', '{}', 0)")
                return
            connection.execute("ALTER TABLE outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
            connection.execute("ALTER TABLE outbox ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0")
            connection.execute("ALTER TABLE outbox ADD COLUMN last_error TEXT")
            if not fifo:
                connection.execute("ALTER TABLE outbox ADD COLUMN event_id TEXT NOT NULL DEFAULT ''")
            for key, event, revision, delivered, attempts, next_attempt, error in (
                ("a:1", "a", 1, 0, 3, 1e20, "ConnectionError"),
                ("a:2", "a", 2, 0, 0, 0, None),
                ("b:1", "b", 1, 0, 1, 0, None),
                ("c:1", "c", 1, 1, 2, 0, None),
            ):
                connection.execute(
                    "INSERT INTO outbox (idempotency_key, payload, delivered, attempts, "
                    "next_attempt_at, last_error) VALUES (?, ?, ?, ?, ?, ?)",
                    (key, json.dumps({"eventId": event, "revision": revision}),
                     delivered, attempts, next_attempt, error),
                )
                if not fifo:
                    connection.execute("UPDATE outbox SET event_id=? WHERE idempotency_key=?", (event, key))

    def test_status_reports_ready_deferred_and_blocked_records_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            self.create_outbox(database)
            before = database.read_bytes()

            status = inspect_outbox(database)

            self.assertEqual(status, {
                "pending": 3,
                "head": {
                    "idempotencyKey": "a:1", "attempts": 3,
                    "nextAttemptAt": 1e20, "errorType": "ConnectionError",
                },
                "readyEvents": 1,
                "deferredEvents": 1,
                "blockedRecords": 1,
                "nextAttemptAt": 0,
                "totalAttempts": 6,
            })
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [database])

    def test_previous_retry_schema_keeps_global_fifo_semantics_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fifo.db"
            self.create_outbox(database, fifo=True)
            before = database.read_bytes()

            status = inspect_outbox(database)

            self.assertEqual(status, {
                "pending": 3,
                "head": {
                    "idempotencyKey": "a:1", "attempts": 3,
                    "nextAttemptAt": 1e20, "errorType": "ConnectionError",
                },
                "schema": "fifo",
                "readyEvents": 0,
                "deferredEvents": 1,
                "blockedRecords": 2,
                "nextAttemptAt": 1e20,
                "totalAttempts": 6,
            })
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [database])

    def test_legacy_status_does_not_migrate_or_create_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.db"
            self.create_outbox(database, legacy=True)
            before = database.read_bytes()

            self.assertEqual(inspect_outbox(database), {"pending": 1, "schema": "legacy"})

            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [database])

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                inspect_outbox(database)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_read_only_connection_closes_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            self.create_outbox(database)
            connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            with patch("campus_safety_ai.apps.deliver_events.sqlite3.connect", return_value=connection) as connect:
                inspect_outbox(database)
            connect.assert_called_once_with(database.resolve().as_uri() + "?mode=ro", uri=True)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_connection_closes_when_status_query_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "invalid.db"
            connection = sqlite3.connect(database)
            with patch("campus_safety_ai.apps.deliver_events.sqlite3.connect", return_value=connection):
                with self.assertRaises(sqlite3.OperationalError):
                    inspect_outbox(database)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_status_needs_only_outbox_and_never_loads_platform_or_sends(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            self.create_outbox(database)
            before = database.read_bytes()
            output = io.StringIO()
            with patch("sys.argv", ["deliver-events", "--status", "--outbox", str(database)]), \
                    patch("campus_safety_ai.apps.deliver_events.load_platform_settings") as settings, \
                    patch("campus_safety_ai.apps.deliver_events.build_platform_destination") as destination, \
                    patch("campus_safety_ai.apps.deliver_events.BackgroundDelivery") as worker, \
                    redirect_stdout(output):
                main()

            self.assertEqual(json.loads(output.getvalue())["pending"], 3)
            settings.assert_not_called()
            destination.assert_not_called()
            worker.assert_not_called()
            self.assertEqual(database.read_bytes(), before)

    def test_status_can_get_outbox_from_platform_config_without_building_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            self.create_outbox(database, legacy=True)
            output = io.StringIO()
            with patch("sys.argv", ["deliver-events", "--status", "--platform-config", "platform.toml"]), \
                    patch("campus_safety_ai.apps.deliver_events.load_platform_settings",
                          return_value=SimpleNamespace(outbox=database)) as settings, \
                    patch("campus_safety_ai.apps.deliver_events.build_platform_destination") as destination, \
                    patch("campus_safety_ai.apps.deliver_events.BackgroundDelivery") as worker, \
                    redirect_stdout(output):
                main()

            self.assertEqual(json.loads(output.getvalue()), {"pending": 1, "schema": "legacy"})
            settings.assert_called_once_with(Path("platform.toml"))
            destination.assert_not_called()
            worker.assert_not_called()

    def test_resume_requires_platform_config_even_with_outbox(self):
        error = io.StringIO()
        with patch("sys.argv", ["deliver-events", "--outbox", "outbox.db"]), \
                patch("campus_safety_ai.apps.deliver_events.load_platform_settings") as settings, \
                patch("campus_safety_ai.apps.deliver_events.BackgroundDelivery") as worker, \
                redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--platform-config is required to resume delivery", error.getvalue())
        settings.assert_not_called()
        worker.assert_not_called()

    def test_status_requires_a_database_location(self):
        error = io.StringIO()
        with patch("sys.argv", ["deliver-events", "--status"]), redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--status requires --outbox or --platform-config", error.getvalue())


if __name__ == "__main__":
    unittest.main()
