import unittest
from datetime import datetime, timezone

from campus_safety_ai.contracts import BBox, Detection, Detections


class ContractTests(unittest.TestCase):
    def test_detection_round_trip_preserves_project_coordinates(self) -> None:
        batch = Detections(
            camera_id="gate-02",
            source_epoch=1,
            sequence=2,
            captured_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
            width=1000,
            height=1000,
            detections=(Detection("person", 0.9, BBox(500, 100, 700, 800)),),
            model_version="fake-v1",
            inference_ms=1.2,
        )
        self.assertEqual(Detections.from_dict(batch.to_dict()), batch)

    def test_bbox_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            BBox(10, 10, 5, 20)


if __name__ == "__main__":
    unittest.main()

