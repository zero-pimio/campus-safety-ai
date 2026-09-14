import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from campus_safety_ai.adapters.easyaiot import (
    EasyAIoTDeliveryError,
    EasyAIoTDestination,
    EasyAIoTEventMapper,
)
from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.event_delivery import EventDelivery


class EasyAIoTAdapterTests(unittest.TestCase):
    def record(self, phase: str = "START") -> EventRecord:
        started = datetime(2026, 8, 25, 8, 0, tzinfo=UTC)
        ended = datetime(2026, 8, 25, 8, 0, 5, tzinfo=UTC) if phase == "END" else None
        return EventRecord(
            schema_version="1.0",
            event_id="evt-1",
            revision=2 if phase == "END" else 1,
            phase=phase,
            event_type="person_fighting",
            severity="critical",
            edge_id="edge-01",
            camera_id="gate-02",
            started_at=started,
            observed_at=ended or started,
            ended_at=ended,
            subject_track_keys=("gate-02:1:1:7",),
            confidence=0.87,
            model_version="fight-v1",
            config_version="fight-v1",
            idempotency_key=f"evt-1:{2 if phase == 'END' else 1}",
            status="CLOSED" if phase == "END" else "OPEN",
            evidence_uris=("runtime/evidence/snapshot.jpg", "runtime/evidence/clip.mp4"),
        )

    def test_mapper_preserves_lifecycle_and_uses_easyaiot_time_format(self) -> None:
        payload = EasyAIoTEventMapper(device_name="东门摄像头").to_payload(self.record("END"))

        self.assertEqual(payload["device_id"], "gate-02")
        self.assertEqual(payload["device_name"], "东门摄像头")
        self.assertEqual(payload["event"], "person_fighting")
        self.assertEqual(payload["task_type"], "realtime")
        self.assertEqual(payload["time"], "2026-08-25 16:00:05")
        self.assertEqual(payload["image_path"], "runtime/evidence/snapshot.jpg")
        self.assertEqual(payload["record_path"], "runtime/evidence/clip.mp4")
        self.assertEqual(payload["correlation_id"], "evt-1:2")
        self.assertEqual(payload["information"]["phase"], "END")
        self.assertEqual(payload["information"]["eventRecord"]["eventId"], "evt-1")

    def test_mapper_does_not_send_a_video_as_an_image_when_snapshot_is_absent(self) -> None:
        record = replace(self.record(), evidence_uris=("runtime/evidence/clip.mp4",))

        payload = EasyAIoTEventMapper().to_payload(record)

        self.assertIsNone(payload["image_path"])
        self.assertEqual(payload["record_path"], "runtime/evidence/clip.mp4")

    def test_destination_posts_json_and_auth_without_touching_core_record(self) -> None:
        requests: list[tuple[str, bytes, dict[str, str], float]] = []

        def poster(endpoint: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
            requests.append((endpoint, body, headers, timeout))
            return 200

        destination = EasyAIoTDestination(
            "http://easyaiot.test/admin-api/video/alert/hook",
            token="secret-token",
            timeout_seconds=3,
            poster=poster,
        )
        record = self.record()
        destination.publish(record.to_dict())

        self.assertEqual(len(requests), 1)
        endpoint, body, headers, timeout = requests[0]
        self.assertEqual(endpoint, "http://easyaiot.test/admin-api/video/alert/hook")
        self.assertEqual(timeout, 3)
        self.assertEqual(headers["Authorization"], "Bearer secret-token")
        self.assertEqual(headers["X-Idempotency-Key"], "evt-1:1")
        payload = json.loads(body)
        self.assertEqual(payload["information"]["eventId"], "evt-1")
        self.assertEqual(payload["time"], "2026-08-25 16:00:00")

    def test_non_success_status_keeps_delivery_failure_explicit(self) -> None:
        destination = EasyAIoTDestination(
            "http://easyaiot.test/alert/hook",
            poster=lambda endpoint, body, headers, timeout: 503,
        )
        with self.assertRaisesRegex(EasyAIoTDeliveryError, "HTTP 503"):
            destination.publish(self.record().to_dict())

    def test_outbox_retains_failed_event_for_a_later_flush(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.sqlite3"
            attempts: list[int] = []

            def failing_poster(endpoint, body, headers, timeout):
                attempts.append(1)
                raise EasyAIoTDeliveryError("temporary outage")

            failed = EasyAIoTDestination("http://easyaiot.test/alert/hook", poster=failing_poster)
            with self.assertRaises(EasyAIoTDeliveryError):
                with EventDelivery(database, failed) as delivery:
                    delivery.submit([self.record()])

            self.assertEqual(len(attempts), 1)
            succeeding = EasyAIoTDestination(
                "http://easyaiot.test/alert/hook",
                poster=lambda endpoint, body, headers, timeout: 200,
            )
            with EventDelivery(database, succeeding) as delivery:
                self.assertEqual(delivery.pending_count(), 1)
                self.assertEqual(delivery.flush(), 1)
                self.assertEqual(delivery.pending_count(), 0)

    def test_invalid_endpoint_is_rejected_before_runtime_delivery(self) -> None:
        with self.assertRaises(ValueError):
            EasyAIoTDestination("localhost:48080/alert/hook")


if __name__ == "__main__":
    unittest.main()
