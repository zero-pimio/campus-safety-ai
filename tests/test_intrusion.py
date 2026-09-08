import unittest
from datetime import datetime, timedelta, UTC

from campus_safety_ai.contracts import BBox, Detection, Detections
from campus_safety_ai.core.event_analysis import EventAnalysis, IntrusionPolicy


class IntrusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started = datetime(2026, 8, 25, tzinfo=UTC)
        self.analysis = EventAnalysis(
            IntrusionPolicy(
                edge_id="edge-01",
                zone_id="right",
                polygon=((0.5, 0), (1, 0), (1, 1), (0.5, 1)),
                enter_seconds=2,
                exit_seconds=1,
                cooldown_seconds=10,
            )
        )

    def batch(self, second: int, bbox: BBox, sequence: int | None = None) -> Detections:
        return Detections(
            camera_id="gate-02",
            source_epoch=1,
            sequence=sequence if sequence is not None else second + 1,
            captured_at=self.started + timedelta(seconds=second),
            width=1000,
            height=1000,
            detections=(Detection("person", 0.9, bbox),),
            model_version="fake-v1",
            inference_ms=1,
        )

    def test_intrusion_uses_real_time_and_emits_start_end(self) -> None:
        inside = BBox(550, 100, 750, 800)
        # The box crosses the boundary while keeping enough overlap for the G1 IoU tracker.
        outside = BBox(295, 100, 695, 800)
        self.assertEqual(self.analysis.advance(self.batch(0, inside)), [])
        self.assertEqual(self.analysis.advance(self.batch(1, inside)), [])
        started = self.analysis.advance(self.batch(2, inside))
        self.assertEqual([record.phase for record in started], ["START"])
        self.assertEqual(self.analysis.advance(self.batch(3, outside)), [])
        ended = self.analysis.advance(self.batch(4, outside))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(started[0].event_id, ended[0].event_id)

    def test_duplicate_or_out_of_order_sequence_is_ignored(self) -> None:
        inside = BBox(550, 100, 750, 800)
        self.analysis.advance(self.batch(0, inside, 1))
        self.assertEqual(self.analysis.advance(self.batch(5, inside, 1)), [])


if __name__ == "__main__":
    unittest.main()
