import unittest
from datetime import datetime, timedelta, timezone

from campus_safety_ai.contracts import BehaviorObservation
from campus_safety_ai.core.fight_analysis import FightEventAnalysis, FightPolicy


class FightFinalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2026, 8, 25, tzinfo=timezone.utc)
        self.analysis = FightEventAnalysis(
            FightPolicy(
                edge_id="edge-01",
                start_score=0.75,
                end_score=0.35,
                confirm_seconds=2,
                clear_seconds=2,
                cooldown_seconds=10,
            )
        )

    def observation(self, second: int, score: float, sequence: int | None = None) -> BehaviorObservation:
        observed = self.origin + timedelta(seconds=second)
        return BehaviorObservation(
            camera_id="gate-02",
            source_epoch=1,
            sequence=sequence if sequence is not None else second + 1,
            observed_at=observed,
            window_started_at=observed - timedelta(seconds=1),
            window_ended_at=observed,
            behavior="fighting",
            score=score,
            model_version="fake-fight-v1",
            subject_track_keys=("gate-02:1:1:7",),
        )

    def test_finalize_closes_event_stuck_in_open_state(self) -> None:
        self.analysis.advance(self.observation(1, 0.9))
        started = self.analysis.advance(self.observation(2, 0.9))
        self.assertEqual([record.phase for record in started], ["START"])

        ended_at = self.origin + timedelta(seconds=30)
        finalized = self.analysis.finalize("gate-02", 1, ended_at)

        self.assertEqual([record.phase for record in finalized], ["END"])
        self.assertEqual(finalized[0].event_id, started[0].event_id)
        self.assertNotEqual(finalized[0].idempotency_key, started[0].idempotency_key)
        self.assertGreater(finalized[0].revision, started[0].revision)
        self.assertEqual(finalized[0].status, "CLOSED")

    def test_finalize_without_open_event_is_a_noop(self) -> None:
        self.analysis.advance(self.observation(1, 0.9))
        self.assertEqual(self.analysis.finalize("gate-02", 1, self.origin), [])
        # Another camera or epoch must not be affected.
        self.analysis.advance(self.observation(1, 0.9))
        self.assertEqual(self.analysis.finalize("other-cam", 2, self.origin), [])

    def test_new_source_epoch_resets_state(self) -> None:
        self.analysis.advance(self.observation(1, 0.9, sequence=1))
        started = self.analysis.advance(self.observation(2, 0.9, sequence=2))

        reopened = self.analysis.advance(
            BehaviorObservation(
                camera_id="gate-02",
                source_epoch=2,
                sequence=1,
                observed_at=self.origin + timedelta(seconds=60),
                window_started_at=self.origin + timedelta(seconds=59),
                window_ended_at=self.origin + timedelta(seconds=60),
                behavior="fighting",
                score=0.9,
                model_version="fake-fight-v1",
            )
        )
        self.assertEqual([record.phase for record in started], ["START"])
        self.assertEqual([record.phase for record in reopened], [])
        self.assertNotEqual(reopened and reopened[0].event_id, started[0].event_id)


if __name__ == "__main__":
    unittest.main()
