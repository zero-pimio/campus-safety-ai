import importlib.util
import unittest
from datetime import datetime, timedelta, UTC


@unittest.skipUnless(
    importlib.util.find_spec("ultralytics") and importlib.util.find_spec("lap"),
    "ByteTrack dependencies not installed",
)
class ByteTrackAdapterTests(unittest.TestCase):
    def test_track_identity_and_expiration_cross_the_project_seam(self) -> None:
        from campus_safety_ai.adapters.runtimes.ultralytics_bytetrack import UltralyticsByteTracker
        from campus_safety_ai.contracts import BBox, Detection, Detections

        origin = datetime(2026, 8, 25, tzinfo=UTC)
        tracker = UltralyticsByteTracker(track_buffer=2)

        def batch(sequence, detections):
            return Detections(
                camera_id="camera-01",
                source_epoch=1,
                sequence=sequence,
                captured_at=origin + timedelta(seconds=sequence),
                width=100,
                height=100,
                detections=tuple(detections),
                model_version="fake",
                inference_ms=0,
            )

        person = Detection("person", 0.9, BBox(10, 10, 30, 80))
        first = tracker.update(batch(1, [person]))
        second = tracker.update(batch(2, [person]))
        expired = ()
        for sequence in range(3, 8):
            expired = tracker.update(batch(sequence, [])).expired_track_keys
            if expired:
                break

        self.assertEqual(first.tracks[0].track_key, second.tracks[0].track_key)
        self.assertEqual(expired, (first.tracks[0].track_key,))

    def test_source_epoch_switch_reports_live_tracks_as_expired(self) -> None:
        from campus_safety_ai.adapters.runtimes.ultralytics_bytetrack import UltralyticsByteTracker
        from campus_safety_ai.contracts import BBox, Detection, Detections

        origin = datetime(2026, 8, 25, tzinfo=UTC)
        tracker = UltralyticsByteTracker(track_buffer=30)

        def batch(sequence, epoch):
            return Detections(
                camera_id="camera-01",
                source_epoch=epoch,
                sequence=sequence,
                captured_at=origin + timedelta(seconds=sequence),
                width=100,
                height=100,
                detections=(Detection("person", 0.9, BBox(10, 10, 30, 80)),),
                model_version="fake",
                inference_ms=0,
            )

        key = tracker.update(batch(1, 1)).tracks[0].track_key
        result = tracker.update(batch(2, 2))

        self.assertEqual(result.expired_track_keys, (key,))
        self.assertNotEqual(result.tracks[0].track_key, key)


if __name__ == "__main__":
    unittest.main()
