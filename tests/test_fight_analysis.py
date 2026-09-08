import unittest
from datetime import datetime, timedelta, UTC

from campus_safety_ai.contracts import BehaviorObservation
from campus_safety_ai.core.fight_analysis import FightEventAnalysis, FightPolicy


class FightAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2026, 8, 25, tzinfo=UTC)
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
            subject_track_keys=("gate-02:1:1:7", "gate-02:1:1:8"),
        )

    def test_continuous_scores_emit_one_start_and_end(self) -> None:
        self.assertEqual(self.analysis.advance(self.observation(1, 0.82)), [])
        started = self.analysis.advance(self.observation(2, 0.87))
        self.assertEqual([record.phase for record in started], ["START"])
        self.assertEqual(started[0].event_type, "person_fighting")
        self.assertEqual(len(started[0].subject_track_keys), 2)

        self.assertEqual(self.analysis.advance(self.observation(3, 0.60)), [])
        self.assertEqual(self.analysis.advance(self.observation(4, 0.20)), [])
        ended = self.analysis.advance(self.observation(5, 0.15))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(started[0].event_id, ended[0].event_id)

    def test_ambiguous_score_breaks_unconfirmed_positive_run(self) -> None:
        self.analysis.advance(self.observation(1, 0.9))
        self.analysis.advance(self.observation(2, 0.5))
        self.assertEqual(self.analysis.advance(self.observation(3, 0.9)), [])

    def test_duplicate_sequence_is_ignored(self) -> None:
        self.analysis.advance(self.observation(1, 0.9, 1))
        self.assertEqual(self.analysis.advance(self.observation(9, 0.9, 1)), [])

    def test_cooldown_blocks_immediate_reopen(self) -> None:
        self.analysis.advance(self.observation(1, 0.9))
        self.analysis.advance(self.observation(2, 0.9))
        self.analysis.advance(self.observation(4, 0.1))
        ended = self.analysis.advance(self.observation(5, 0.1))
        self.assertEqual([record.phase for record in ended], ["END"])

        self.assertEqual(self.analysis.advance(self.observation(6, 0.9)), [])
        self.assertEqual(self.analysis.advance(self.observation(7, 0.9)), [])
        self.assertEqual(self.analysis.advance(self.observation(16, 0.9)), [])
        reopened = self.analysis.advance(self.observation(17, 0.9))
        self.assertEqual([record.phase for record in reopened], ["START"])


if __name__ == "__main__":
    unittest.main()
