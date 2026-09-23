import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from campus_safety_ai.core.fall_analysis import FallEventAnalysis, FallObservation, FallPolicy


class FallAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2026, 9, 23, tzinfo=UTC)
        self.policy = FallPolicy(
            edge_id="edge-test", confirm_seconds=2, clear_seconds=2,
            cooldown_seconds=3, max_observation_gap_seconds=1,
        )
        self.analysis = FallEventAnalysis(self.policy)

    def observation(self, second: float, score: float | None, **overrides) -> FallObservation:
        at = self.origin + timedelta(seconds=second)
        observation = FallObservation(
            camera_id="gate", source_epoch=1, sequence=round(second * 100) + 1,
            observed_at=at, window_started_at=at - timedelta(seconds=5),
            window_ended_at=at, score=score, model_version="fall-test",
            subject_track_keys=("gate:1:track-1",),
        )
        return replace(observation, **overrides)

    def start(self) -> list:
        self.analysis.advance(self.observation(0, 0.9))
        self.analysis.advance(self.observation(1, 0.9))
        return self.analysis.advance(self.observation(2, 0.9))

    def test_normal_fall_motion_clear_cooldown_and_second_event(self) -> None:
        records = []
        scores = [0.1, 0.9, 0.9, 0.9, 0.5, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
        for second, score in enumerate(scores):
            records.extend(self.analysis.advance(self.observation(second, score)))
        self.assertEqual([item.phase for item in records], ["START", "END", "START"])
        first, ended, second = records
        self.assertEqual(first.event_type, "person_falling")
        self.assertEqual(first.started_at, self.origin + timedelta(seconds=1))
        self.assertEqual(first.observed_at, self.origin + timedelta(seconds=3))
        self.assertEqual(ended.observed_at, self.origin + timedelta(seconds=7))
        self.assertEqual(second.observed_at, self.origin + timedelta(seconds=12))
        self.assertEqual(first.event_id, ended.event_id)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertEqual((first.revision, ended.revision, second.revision), (1, 2, 1))
        self.assertEqual(len({item.idempotency_key for item in records}), 3)

    def test_long_overlapping_windows_do_not_confirm_immediately(self) -> None:
        self.assertEqual(self.analysis.advance(self.observation(0, 0.95)), [])
        self.assertEqual(self.analysis.advance(self.observation(1, 0.95)), [])
        records = self.analysis.advance(self.observation(2, 0.95))
        self.assertEqual([item.phase for item in records], ["START"])
        self.assertEqual(records[0].started_at, self.origin)

    def test_unknown_breaks_confirmation_and_clear_without_implying_recovery(self) -> None:
        self.analysis.advance(self.observation(0, 0.9))
        self.analysis.advance(self.observation(1, 0.9))
        self.assertEqual(self.analysis.advance(self.observation(2, None, reason="missing")), [])
        self.assertEqual(self.analysis.advance(self.observation(3, 0.9)), [])
        self.assertEqual(self.analysis.advance(self.observation(4, 0.9)), [])
        started = self.analysis.advance(self.observation(5, 0.9))
        self.assertEqual([item.phase for item in started], ["START"])
        self.analysis.advance(self.observation(6, 0.1))
        self.assertEqual(self.analysis.advance(self.observation(7, None, reason="missing")), [])
        self.assertEqual(self.analysis.advance(self.observation(8, 0.1)), [])
        self.assertEqual(self.analysis.advance(self.observation(9, 0.1)), [])
        ended = self.analysis.advance(self.observation(10, 0.1))
        self.assertEqual([item.phase for item in ended], ["END"])
        self.assertEqual(self.analysis.last_record_reasons[ended[0].idempotency_key], "motion_cleared")

    def test_hysteresis_band_breaks_each_continuous_run(self) -> None:
        self.analysis.advance(self.observation(0, 0.9))
        self.analysis.advance(self.observation(1, 0.5))
        self.analysis.advance(self.observation(2, 0.9))
        self.assertEqual(self.analysis.advance(self.observation(3, 0.9)), [])
        started = self.analysis.advance(self.observation(4, 0.9))
        self.assertEqual([item.phase for item in started], ["START"])
        self.analysis.advance(self.observation(5, 0.1))
        self.analysis.advance(self.observation(6, 0.5))
        self.analysis.advance(self.observation(7, 0.1))
        self.assertEqual(self.analysis.advance(self.observation(8, 0.1)), [])

    def test_large_gap_breaks_candidate_and_closes_open_event_as_boundary(self) -> None:
        self.analysis.advance(self.observation(0, 0.9))
        self.assertEqual(self.analysis.advance(self.observation(10, 0.9)), [])
        self.analysis.advance(self.observation(11, 0.9))
        started = self.analysis.advance(self.observation(12, 0.9))
        self.assertEqual([item.phase for item in started], ["START"])
        ended = self.analysis.advance(self.observation(20, None, reason="gap"))
        self.assertEqual([item.phase for item in ended], ["END"])
        self.assertEqual(ended[0].ended_at, self.origin + timedelta(seconds=12))
        self.assertEqual(ended[0].observed_at, self.origin + timedelta(seconds=20))
        self.assertEqual(ended[0].confidence, 0.9)
        self.assertEqual(self.analysis.last_reason, "observation_gap")

    def test_duplicate_and_out_of_order_input_cannot_advance_lifecycle(self) -> None:
        self.analysis.advance(self.observation(0, 0.9))
        self.analysis.advance(self.observation(1, 0.9))
        self.assertEqual(self.analysis.advance(self.observation(20, 0.9, sequence=101)), [])
        self.assertEqual(self.analysis.advance(self.observation(0.5, 0.1, sequence=900)), [])
        started = self.analysis.advance(self.observation(2, 0.9))
        self.assertEqual([item.phase for item in started], ["START"])
        self.assertEqual(self.analysis.advance(self.observation(2, 0.1, sequence=901)), [])
        self.assertEqual(self.analysis.advance(self.observation(2, 0.9)), [])
        self.assertEqual(self.analysis.last_record_reasons, {})

    def test_track_change_resets_candidate_and_closes_old_subject(self) -> None:
        self.analysis.advance(self.observation(0, 0.9))
        self.analysis.advance(self.observation(1, 0.9))
        new_track = {"subject_track_keys": ("gate:1:track-2",)}
        self.assertEqual(self.analysis.advance(self.observation(2, 0.9, **new_track)), [])
        self.analysis.advance(self.observation(3, 0.9, **new_track))
        started = self.analysis.advance(self.observation(4, 0.9, **new_track))
        self.assertEqual(started[0].subject_track_keys, ("gate:1:track-2",))
        ended = self.analysis.advance(self.observation(5, None, reason="track_change"))
        self.assertEqual([item.phase for item in ended], ["END"])
        self.assertEqual(ended[0].subject_track_keys, ("gate:1:track-2",))
        self.assertEqual(self.analysis.last_reason, "track_changed")
        self.assertEqual(ended[0].model_version, "fall-test")

    def test_missing_or_multiple_subjects_are_unknown(self) -> None:
        self.assertEqual(self.analysis.advance(self.observation(0, 1.0, subject_track_keys=())), [])
        self.assertEqual(self.analysis.advance(self.observation(1, 1.0, subject_track_keys=("a", "b"))), [])
        self.analysis.advance(self.observation(2, 0.9))
        self.analysis.advance(self.observation(3, 0.9))
        started = self.analysis.advance(self.observation(4, 0.9))
        self.assertEqual([item.phase for item in started], ["START"])
        for second in (5, 6, 7):
            self.assertEqual(self.analysis.advance(self.observation(second, 0.0, subject_track_keys=())), [])
        ended = self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=8), "source_disconnected")
        self.assertEqual([item.phase for item in ended], ["END"])
        self.assertEqual(ended[0].confidence, 0.9)
        self.assertEqual(self.analysis.last_reason, "source_disconnected")

    def test_epoch_change_closes_old_and_never_accepts_delayed_old_epoch(self) -> None:
        started = self.start()
        ended = self.analysis.advance(self.observation(3, 0.9, source_epoch=2, sequence=1))
        self.assertEqual([item.phase for item in ended], ["END"])
        self.assertEqual(started[0].event_id, ended[0].event_id)
        self.assertEqual(self.analysis.last_reason, "source_epoch_changed")
        self.assertEqual(self.analysis.advance(self.observation(4, 0.1, source_epoch=1, sequence=999)), [])
        self.assertEqual(self.analysis.advance(self.observation(4, 0.9, source_epoch=2, sequence=2)), [])
        restarted = self.analysis.advance(self.observation(5, 0.9, source_epoch=2, sequence=3))
        self.assertEqual([item.phase for item in restarted], ["START"])
        self.assertNotEqual(restarted[0].event_id, started[0].event_id)
        self.assertEqual(self.analysis.state_count, 1)

    def test_two_cameras_keep_independent_runs(self) -> None:
        for second in (0, 1):
            self.analysis.advance(self.observation(second, 0.9))
            self.analysis.advance(self.observation(second, 0.1, camera_id="hall"))
        started = self.analysis.advance(self.observation(2, 0.9))
        self.assertEqual([item.camera_id for item in started], ["gate"])
        self.assertEqual(self.analysis.advance(self.observation(2, 0.9, camera_id="hall")), [])
        self.assertEqual(self.analysis.state_count, 2)

    def test_finalize_closes_once_cleans_state_and_retains_epoch_watermark(self) -> None:
        self.start()
        closed = self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=3), "eof")
        self.assertEqual([item.phase for item in closed], ["END"])
        self.assertEqual(self.analysis.last_record_reasons, {closed[0].idempotency_key: "eof"})
        self.assertEqual(self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=3)), [])
        self.assertEqual(self.analysis.advance(self.observation(4, 0.9)), [])
        self.assertEqual(self.analysis.state_count, 1)
        self.analysis.release_camera("gate")
        self.assertEqual(self.analysis.state_count, 0)

    def test_state_capacity_and_evidence_are_bounded(self) -> None:
        analysis = FallEventAnalysis(replace(self.policy, max_cameras=1, max_evidence_uris=2, confirm_seconds=0))
        analysis.advance(self.observation(0, 0.9, evidence_uris=("0.jpg",)))
        for second in range(1, 20):
            analysis.advance(self.observation(second, 0.9, evidence_uris=(f"{second}.jpg",)))
        with self.assertRaisesRegex(ValueError, "camera limit"):
            analysis.advance(self.observation(20, 0.9, camera_id="another"))
        with self.assertRaisesRegex(ValueError, "finalize"):
            analysis.release_camera("gate")
        closed = analysis.finalize("gate", 1, self.origin + timedelta(seconds=20))
        self.assertEqual(closed[0].evidence_uris, ("18.jpg", "19.jpg"))
        analysis.release_camera("gate")
        analysis.advance(self.observation(20, 0.9, camera_id="another"))
        self.assertEqual(analysis.state_count, 1)

    def test_unknown_round_trips_and_invalid_values_fail(self) -> None:
        observation = self.observation(0, None, reason="warming_up")
        self.assertEqual(FallObservation.from_dict(observation.to_dict()), observation)
        for score in (-0.1, 1.1, float("nan"), float("inf"), True, "0.9"):
            with self.assertRaises(ValueError):
                self.observation(0, score)
        for changes in ({"clear_seconds": float("nan")}, {"confirm_seconds": -1},
                        {"max_observation_gap_seconds": 0}, {"max_cameras": 0},
                        {"max_cameras": 1.5}, {"max_cameras": True},
                        {"confirm_seconds": True}, {"start_score": True}):
            with self.assertRaises(ValueError):
                replace(self.policy, **changes)
        for changes in ({"camera_id": "  "}, {"source_epoch": True}, {"source_epoch": 1.0},
                        {"sequence": True}, {"sequence": 1.0}, {"subject_track_keys": ("",)}):
            with self.assertRaises(ValueError):
                replace(observation, **changes)
        for changes in ({"cameraId": None}, {"sourceEpoch": 1.5}, {"sequence": True},
                        {"score": True}, {"score": "0.9"}, {"subjectTrackKeys": "a"}):
            with self.assertRaises(ValueError):
                FallObservation.from_dict({**observation.to_dict(), **changes})


if __name__ == "__main__":
    unittest.main()
