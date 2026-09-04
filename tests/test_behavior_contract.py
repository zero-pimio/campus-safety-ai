import unittest
from datetime import datetime, timedelta, timezone

from campus_safety_ai.contracts import BehaviorObservation


class BehaviorContractTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        observed = datetime(2026, 8, 25, tzinfo=timezone.utc)
        value = BehaviorObservation(
            camera_id="gate-02",
            source_epoch=1,
            sequence=3,
            observed_at=observed,
            window_started_at=observed - timedelta(seconds=2),
            window_ended_at=observed,
            behavior="fighting",
            score=0.91,
            model_version="fight-v1",
            subject_track_keys=("track-1", "track-2"),
            evidence_uris=("snapshot.jpg", "clip.mp4"),
        )
        self.assertEqual(BehaviorObservation.from_dict(value.to_dict()), value)

    def test_rejects_score_outside_probability_range(self) -> None:
        observed = datetime(2026, 8, 25, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            BehaviorObservation(
                camera_id="gate-02",
                source_epoch=1,
                sequence=1,
                observed_at=observed,
                window_started_at=observed,
                window_ended_at=observed,
                behavior="fighting",
                score=1.1,
                model_version="fight-v1",
            )


if __name__ == "__main__":
    unittest.main()
