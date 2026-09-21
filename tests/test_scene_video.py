from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from campus_safety_ai.apps import scene_video
from campus_safety_ai.apps.offline_replay import run as replay_detections
from campus_safety_ai.contracts import BBox, Detection, Detections
from campus_safety_ai.core.event_analysis import EventAnalysis, SimpleIoUTracker
from campus_safety_ai.core.parking_analysis import ParkingAnalysis
from campus_safety_ai.settings import load_intrusion_policy, load_parking_policy


class FakePerception:
    def __init__(self, label: str, fail_sequence: int | None = None, failure: BaseException | None = None):
        self.label = label
        self.fail_sequence = fail_sequence
        self.failure = failure

    def detect(self, frame):
        if frame.sequence == self.fail_sequence:
            assert self.failure is not None
            raise self.failure
        return Detections(
            frame.camera_id, frame.source_epoch, frame.sequence, frame.captured_at,
            frame.width, frame.height, (Detection(self.label, 0.9, BBox(50, 20, 110, 90)),),
            "integration-detector", 1.0,
        )


class SceneVideoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("optional OpenCV and NumPy extras are not installed") from None
        cls.cv2 = cv2
        cls.np = np

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / "source.avi"
        writer = self.cv2.VideoWriter(str(self.video), self.cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (160, 120))
        if not writer.isOpened():
            self.skipTest("MJPG video writer is unavailable")
        try:
            for index in range(12):
                frame = self.np.full((120, 160, 3), 30 + index * 8, dtype=self.np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        self.detector = self.root / "local-model.pt"
        self.detector.write_bytes(b"model construction is patched; model download is never required")
        self.scene_config = self.root / "scene.toml"
        self.scene_config.write_text(
            'schema_version = "1.0"\n'
            '[intrusion_zone]\nzone_id = "test-intrusion-zone"\n'
            'polygon_normalized = [[0.2,0.1],[0.9,0.1],[0.9,0.95],[0.2,0.95]]\n'
            '[parking_zone]\nzone_id = "test-parking-zone"\n'
            'polygon_normalized = [[0.2,0.1],[0.9,0.1],[0.9,0.95],[0.2,0.95]]\n',
            encoding="utf-8",
        )
        self.event_configs = {}
        for kind in ("intrusion", "parking"):
            path = self.root / f"{kind}.toml"
            shared = (
                'schema_version = "1.0"\nedge_id = "integration-edge"\n'
                f'config_version = "integration-{kind}"\nminimum_confidence = 0.4\n'
                'exit_seconds = 0.4\ncooldown_seconds = 2.0\nmax_observation_gap_seconds = 1.0\n'
            )
            policy = (
                'target_labels = ["person"]\nevent_type = "perimeter_test"\nenter_seconds = 0.4\n'
                if kind == "intrusion" else
                'target_labels = ["truck"]\nstationary_seconds = 0.4\nmovement_threshold = 0.02\n'
            )
            path.write_text(shared + policy, encoding="utf-8")
            self.event_configs[kind] = path
        self.origin = datetime(2026, 9, 21, tzinfo=UTC)

    def run_video(self, output: Path, kind: str = "intrusion", *, perception=None, **kwargs):
        perception = perception or FakePerception("person" if kind == "intrusion" else "truck")
        arguments = {
            "event_kind": kind, "event_config": self.event_configs[kind], "scene_config": self.scene_config,
            "detector": self.detector, "camera_id": "test-camera", "source_epoch": 7,
            "started_at": self.origin, "frame_stride": 1, "tracker_backend": "simple_iou",
            "device": "cpu", "annotated_video": False,
        }
        with patch("campus_safety_ai.apps.scene_video.UltralyticsPerception", return_value=perception) as factory:
            report = scene_video.run(self.video, output, **{**arguments, **kwargs})
        factory.assert_called_once_with(str(self.detector), self.detector.stem, confidence=0.1, device="cpu")
        return report

    @staticmethod
    def jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def assert_delivery(self, output, events):
        with sqlite3.connect(output / "outbox.sqlite3") as connection:
            rows = connection.execute("SELECT payload, delivered FROM outbox ORDER BY rowid").fetchall()
        self.assertEqual([json.loads(payload) for payload, _ in rows], events)
        self.assertEqual([delivered for _, delivered in rows], [1] * len(events))

    def test_intrusion_and_parking_write_complete_artifacts_and_replay_identical_lifecycle(self):
        for kind, expected_type in (("intrusion", "perimeter_test"), ("parking", "illegal_parking")):
            with self.subTest(kind=kind):
                output = self.root / kind
                report = self.run_video(output, kind)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["finish_reason"], "eof")
                self.assertEqual(report["processed_frames"], 12)
                self.assertEqual(report["decoded_frames"], 12)
                self.assertEqual((report["starts"], report["ends"], report["open_events_at_exit"]), (1, 1, 0))
                self.assertEqual(report, json.loads((output / "run.json").read_text(encoding="utf-8")))
                self.assertEqual(report["evidence_errors"], [])
                for field in ("source_sha256", "detector_sha256", "event_config_sha256", "scene_config_sha256"):
                    self.assertEqual(len(report[field]), 64)
                detections = self.jsonl(output / "detections.jsonl")
                self.assertEqual(len(detections), 12)
                self.assertEqual(detections[-1]["sequence"], 12)
                self.assertEqual(detections[-1]["capturedAt"], "2026-09-21T00:00:02.200000Z")
                events = self.jsonl(output / "events.jsonl")
                self.assertEqual([event["phase"] for event in events], ["START", "END"])
                self.assertEqual(events[0]["eventType"], expected_type)
                self.assertEqual(events[0]["edgeId"], "integration-edge")
                self.assertEqual(events[0]["configVersion"], f"integration-{kind}")
                self.assertEqual(events[0]["eventId"], events[1]["eventId"])
                self.assertNotEqual(events[0]["idempotencyKey"], events[1]["idempotencyKey"])
                for event in events:
                    self.assertEqual(len(event["evidenceUris"]), 1)
                    snapshot = Path(event["evidenceUris"][0])
                    self.assertTrue(snapshot.is_absolute())
                    image = self.cv2.imread(str(snapshot))
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape[:2], (120, 160))
                self.assert_delivery(output, events)

                policy = (load_intrusion_policy if kind == "intrusion" else load_parking_policy)(
                    self.event_configs[kind], self.scene_config
                )
                analysis = (EventAnalysis if kind == "intrusion" else ParkingAnalysis)(policy, SimpleIoUTracker())
                replay_path = self.root / f"{kind}-replayed.jsonl"
                self.assertEqual(replay_detections(output / "detections.jsonl", replay_path, analysis), 2)
                lifecycle = [{key: value for key, value in event.items() if key != "evidenceUris"} for event in events]
                replayed = [{key: value for key, value in event.items() if key != "evidenceUris"}
                            for event in self.jsonl(replay_path)]
                self.assertEqual(replayed, lifecycle)

    def test_annotated_video_contains_every_processed_frame(self):
        output = self.root / "annotated"
        report = self.run_video(output, annotated_video=True)
        capture = self.cv2.VideoCapture(str(output / "annotated.mp4"))
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(self.cv2.CAP_PROP_FPS), 5)
            count = 0
            while True:
                available, frame = capture.read()
                if not available:
                    break
                self.assertEqual(frame.shape[:2], (184, 160))
                count += 1
        finally:
            capture.release()
        self.assertEqual(count, report["processed_frames"])

    def test_inference_failure_records_failure_and_delivers_closure(self):
        output = self.root / "failed"
        failure = RuntimeError("simulated inference failure")
        with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
            self.run_video(output, perception=FakePerception("person", fail_sequence=4, failure=failure))
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["finish_reason"], "error")
        self.assertEqual(report["processed_frames"], 3)
        self.assertEqual(report["open_events_at_exit"], 0)
        self.assertIn("simulated inference failure", report["error"])
        events = self.jsonl(output / "events.jsonl")
        self.assertEqual([event["phase"] for event in events], ["START", "END"])
        self.assertEqual(events[-1]["endedAt"], "2026-09-21T00:00:00.400000Z")
        self.assert_delivery(output, events)

    def test_snapshot_failure_is_reported_without_suppressing_events(self):
        output = self.root / "snapshot-failed"
        with patch("cv2.imwrite", return_value=False):
            report = self.run_video(output)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["evidence_errors"]), 2)
        events = self.jsonl(output / "events.jsonl")
        self.assertEqual([event["phase"] for event in events], ["START", "END"])
        self.assertTrue(all(not event["evidenceUris"] for event in events))
        self.assert_delivery(output, events)

    def test_existing_nonempty_output_is_preserved_before_loading_model(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "run.json"
        sentinel.write_text("existing experiment", encoding="utf-8")
        with patch("campus_safety_ai.apps.scene_video.UltralyticsPerception") as factory:
            with self.assertRaisesRegex(ValueError, "empty"):
                scene_video.run(
                    self.video, output, event_kind="intrusion", event_config=self.event_configs["intrusion"],
                    scene_config=self.scene_config, detector=self.detector, camera_id="test-camera",
                )
        factory.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing experiment")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_main_loads_explicit_parking_config_and_disables_video(self):
        output = self.root / "cli-parking"
        argv = [
            "campus-safety-scene-video", "--video", str(self.video), "--output-dir", str(output),
            "--event", "parking", "--event-config", str(self.event_configs["parking"]),
            "--scene-config", str(self.scene_config), "--detector", str(self.detector),
            "--camera-id", "cli-camera", "--source-epoch", "3", "--frame-stride", "1",
            "--tracker", "simple_iou", "--no-video", "--started-at", self.origin.isoformat(),
        ]
        with (
            patch("sys.argv", argv),
            patch("campus_safety_ai.apps.scene_video.UltralyticsPerception", return_value=FakePerception("truck")),
            redirect_stdout(io.StringIO()) as printed,
        ):
            scene_video.main()
        self.assertIn("completed", printed.getvalue())
        self.assertFalse((output / "annotated.mp4").exists())
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["camera_id"], "cli-camera")
        self.assertEqual(report["source_epoch"], 3)
        self.assertEqual(report["event_kind"], "parking")
        self.assertEqual((report["starts"], report["ends"]), (1, 1))

    def test_model_initialization_failure_leaves_failed_run_report(self):
        output = self.root / "initialization-failed"
        with patch("campus_safety_ai.apps.scene_video.UltralyticsPerception", side_effect=RuntimeError("bad weights")):
            with self.assertRaisesRegex(RuntimeError, "bad weights"):
                scene_video.run(
                    self.video, output, event_kind="intrusion", event_config=self.event_configs["intrusion"],
                    scene_config=self.scene_config, detector=self.detector, camera_id="test-camera", frame_stride=1,
                    tracker_backend="simple_iou", started_at=self.origin, annotated_video=False,
                )
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_frames"], 0)
        self.assertIn("bad weights", report["error"])

    def test_keyboard_interrupt_records_interruption_and_closes_open_event(self):
        output = self.root / "interrupted"
        with self.assertRaises(KeyboardInterrupt):
            self.run_video(output, perception=FakePerception("person", fail_sequence=4, failure=KeyboardInterrupt()))
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["open_events_at_exit"], 0)
        events = self.jsonl(output / "events.jsonl")
        self.assertEqual([event["phase"] for event in events], ["START", "END"])
        self.assert_delivery(output, events)


if __name__ == "__main__":
    unittest.main()
