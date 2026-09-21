import math
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from campus_safety_ai.contracts import BBox, Detection, Detections, Track
from campus_safety_ai.core.event_analysis import TrackingResult
from campus_safety_ai.core.parking_analysis import ParkingAnalysis, ParkingPolicy


class FixedTracker:
    def __init__(self):
        self.expired = ()
        self.calls = 0

    def update(self, batch):
        self.calls += 1
        tracks = tuple(Track(f"{batch.camera_id}:{batch.source_epoch}:1:{index}", detection.label,
                             detection.confidence, detection.bbox, batch.captured_at)
                       for index, detection in enumerate(batch.detections, 1))
        expired, self.expired = self.expired, ()
        return TrackingResult(tracks, expired)


class ParkingAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2026, 9, 21, tzinfo=UTC)
        self.policy = ParkingPolicy("edge-1", "no-parking", ((.2, .1), (.9, .1), (.9, .9), (.2, .9)),
                                    stationary_seconds=3, exit_seconds=2, cooldown_seconds=3,
                                    max_observation_gap_seconds=2, movement_threshold=.02)
        self.tracker = FixedTracker()
        self.analysis = ParkingAnalysis(self.policy, self.tracker)

    def batch(self, second, x=.4, *, sequence=None, epoch=1, camera="gate", label="car", confidence=.9):
        detections = () if x is None else (Detection(label, confidence, BBox(x * 1000 - 50, 400,
                                                                           x * 1000 + 50, 600)),)
        return Detections(camera, epoch, int(second) + 1 if sequence is None else sequence,
                          self.origin + timedelta(seconds=second), 1000, 1000, detections, "vehicle-test", 1)

    def start(self, analysis=None, **kwargs):
        analysis = analysis or self.analysis
        self.assertEqual(analysis.advance(self.batch(0, **kwargs)), [])
        self.assertEqual(analysis.advance(self.batch(1, **kwargs)), [])
        self.assertEqual(analysis.advance(self.batch(2, **kwargs)), [])
        records = analysis.advance(self.batch(3, **kwargs))
        self.assertEqual([record.phase for record in records], ["START"])
        return records[0]

    def test_stationary_vehicle_starts_once_and_eof_closes_same_event(self):
        started = self.start()
        self.assertEqual(started.event_type, "illegal_parking")
        self.assertEqual(started.started_at, self.origin)
        self.assertEqual(started.observed_at, self.origin + timedelta(seconds=3))
        self.assertEqual(self.analysis.advance(self.batch(4)), [])
        ended = self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=5))
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0].event_id, started.event_id)
        self.assertEqual(ended[0].revision, 2)
        self.assertNotEqual(ended[0].idempotency_key, started.idempotency_key)
        self.assertEqual(ended[0].status, "CLOSED")
        self.assertEqual(self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=6)), [])
        self.assertEqual(self.analysis._states, {})
        self.assertEqual(self.analysis.advance(self.batch(7)), [])

    def test_slow_cumulative_motion_does_not_count_as_stationary(self):
        for second in range(24):
            # Every adjacent displacement is below .02; cumulative drift is not.
            self.assertEqual(self.analysis.advance(self.batch(second, .3 + second * .009)), [])

    def test_short_stop_and_nonvehicle_never_open(self):
        self.analysis.advance(self.batch(0))
        self.analysis.advance(self.batch(1))
        for second in range(2, 7):
            self.assertEqual(self.analysis.advance(self.batch(second, .1)), [])
        self.assertEqual(self.analysis._states, {})
        other = ParkingAnalysis(self.policy, FixedTracker())
        for second in range(5):
            self.assertEqual(other.advance(self.batch(second, label="person")), [])
        self.assertEqual(other._states, {})

    def test_tracker_receives_only_vehicle_labels_including_low_confidence(self):
        class CapturingTracker(FixedTracker):
            def update(self, batch):
                self.last_batch = batch
                return super().update(batch)

        tracker = CapturingTracker()
        analysis = ParkingAnalysis(self.policy, tracker)
        person = Detection("person", .99, BBox(350, 400, 450, 600))
        car = Detection("car", .1, BBox(350, 400, 450, 600))
        for second in range(5):
            self.assertEqual(analysis.advance(replace(self.batch(second), detections=(person, car))), [])
            self.assertEqual(tracker.last_batch.detections, (car,))
        self.assertEqual(analysis._states, {})

    def test_move_or_leave_requires_continuous_exit_time(self):
        for position in (.5, .1):
            with self.subTest(position=position):
                analysis = ParkingAnalysis(self.policy, FixedTracker())
                started = self.start(analysis)
                self.assertEqual(analysis.advance(self.batch(4, position)), [])
                self.assertEqual(analysis.advance(self.batch(5, position)), [])
                ended = analysis.advance(self.batch(6, position))
                self.assertEqual([record.phase for record in ended], ["END"])
                self.assertEqual(ended[0].event_id, started.event_id)

    def test_return_to_anchor_cancels_pending_end(self):
        self.start()
        self.assertEqual(self.analysis.advance(self.batch(4, .5)), [])
        self.assertEqual(self.analysis.advance(self.batch(5, .4)), [])
        self.assertEqual(self.analysis.advance(self.batch(6, .5)), [])
        self.assertEqual(self.analysis.advance(self.batch(7, .5)), [])
        self.assertEqual([r.phase for r in self.analysis.advance(self.batch(8, .5))], ["END"])

    def test_short_missing_or_low_confidence_observation_resets_dwell(self):
        for missing in (self.batch(2, None), self.batch(2, confidence=.1)):
            with self.subTest(missing=missing.detections):
                analysis = ParkingAnalysis(self.policy, FixedTracker())
                analysis.advance(self.batch(0))
                analysis.advance(self.batch(1))
                self.assertEqual(analysis.advance(missing), [])
                for second in (3, 4, 5):
                    self.assertEqual(analysis.advance(self.batch(second)), [])
                started = analysis.advance(self.batch(6))
                self.assertEqual([r.phase for r in started], ["START"])
                self.assertEqual(started[0].started_at, self.origin + timedelta(seconds=3))

    def test_active_event_survives_short_missing_observation(self):
        started = self.start()
        self.assertEqual(self.analysis.advance(self.batch(4, None)), [])
        self.assertEqual(self.analysis.advance(self.batch(5)), [])
        ended = self.analysis.finalize("gate", 1, self.origin + timedelta(seconds=6))
        self.assertEqual(ended[0].event_id, started.event_id)

    def test_missing_observation_does_not_confirm_departure(self):
        self.start()
        self.analysis.advance(self.batch(4, .5))
        self.assertEqual(self.analysis.advance(self.batch(5, None)), [])
        self.assertEqual(self.analysis.advance(self.batch(6, .5)), [])
        self.assertEqual(self.analysis.advance(self.batch(7, .5)), [])
        self.assertEqual([r.phase for r in self.analysis.advance(self.batch(8, .5))], ["END"])

    def test_long_gap_closes_open_event_and_restarts_candidate_even_if_tracker_reuses_key(self):
        first = self.start()
        ended = self.analysis.advance(self.batch(7))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].event_id, first.event_id)
        for second in (8, 9):
            self.assertEqual(self.analysis.advance(self.batch(second)), [])
        second = self.analysis.advance(self.batch(10))
        self.assertEqual([record.phase for record in second], ["START"])
        self.assertEqual(second[0].started_at, self.origin + timedelta(seconds=7))
        self.assertNotEqual(second[0].event_id, first.event_id)

    def test_long_gap_before_start_does_not_accumulate_unobserved_time(self):
        self.analysis.advance(self.batch(0))
        self.analysis.advance(self.batch(1))
        for second in (5, 6, 7):
            self.assertEqual(self.analysis.advance(self.batch(second)), [])
        started = self.analysis.advance(self.batch(8))
        self.assertEqual(started[0].started_at, self.origin + timedelta(seconds=5))

    def test_expired_track_closes_and_removes_state(self):
        started = self.start()
        self.tracker.expired = started.subject_track_keys
        records = self.analysis.advance(self.batch(4, None))
        self.assertEqual([record.phase for record in records], ["END"])
        self.assertEqual(records[0].camera_id, "gate")
        self.assertEqual(self.analysis._states, {})

    def test_source_epoch_closes_then_rejects_late_old_epoch(self):
        started = self.start()
        ended = self.analysis.advance(self.batch(4, epoch=2, sequence=0))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].event_id, started.event_id)
        calls = self.tracker.calls
        self.assertEqual(self.analysis.advance(self.batch(20, epoch=1, sequence=100)), [])
        self.assertEqual(self.tracker.calls, calls)
        for second in (5, 6):
            self.assertEqual(self.analysis.advance(self.batch(second, epoch=2)), [])
        new = self.analysis.advance(self.batch(7, epoch=2))
        self.assertEqual(new[0].subject_track_keys, ("gate:2:1:1",))

    def test_duplicates_and_backward_timestamps_never_advance_tracker(self):
        self.analysis.advance(self.batch(0))
        self.analysis.advance(self.batch(1))
        for invalid in (self.batch(20, sequence=2), self.batch(.5, sequence=3), self.batch(1, sequence=3)):
            self.assertEqual(self.analysis.advance(invalid), [])
        self.assertEqual(self.tracker.calls, 2)
        self.assertEqual(self.analysis.advance(self.batch(2, sequence=3)), [])
        self.assertEqual([r.phase for r in self.analysis.advance(self.batch(3, sequence=4))], ["START"])

    def test_cooldown_requires_fresh_dwell_after_expiry(self):
        first = self.start()
        for second in (4, 5):
            self.analysis.advance(self.batch(second, .5))
        ended = self.analysis.advance(self.batch(6, .5))
        self.assertEqual([r.phase for r in ended], ["END"])
        for second in range(7, 12):
            self.assertEqual(self.analysis.advance(self.batch(second, .5)), [])
        second = self.analysis.advance(self.batch(12, .5))
        self.assertEqual([r.phase for r in second], ["START"])
        self.assertEqual(second[0].started_at, self.origin + timedelta(seconds=9))
        self.assertNotEqual(second[0].event_id, first.event_id)

    def test_low_confidence_gap_does_not_erase_normal_end_cooldown(self):
        analysis = ParkingAnalysis(replace(self.policy, exit_seconds=1, cooldown_seconds=30), FixedTracker())
        self.start(analysis)
        analysis.advance(self.batch(4, .5))
        self.assertEqual([r.phase for r in analysis.advance(self.batch(5, .5))], ["END"])
        for second in (6, 7, 8):
            self.assertEqual(analysis.advance(self.batch(second, .5, confidence=.1)), [])
        for second in range(9, 38):
            self.assertEqual(analysis.advance(self.batch(second, .5)), [])
        started = analysis.advance(self.batch(38, .5))
        self.assertEqual([r.phase for r in started], ["START"])
        self.assertEqual(started[0].started_at, self.origin + timedelta(seconds=35))

    def test_retained_cooldown_is_removed_at_expiry_or_explicit_track_expiration(self):
        for explicitly_expired in (False, True):
            with self.subTest(explicitly_expired=explicitly_expired):
                tracker = FixedTracker()
                analysis = ParkingAnalysis(replace(self.policy, exit_seconds=1, cooldown_seconds=30), tracker)
                started = self.start(analysis)
                analysis.advance(self.batch(4, .5))
                analysis.advance(self.batch(5, .5))
                for second in range(6, 10):
                    analysis.advance(self.batch(second, .5, confidence=.1))
                self.assertEqual(len(analysis._states), 1)
                if explicitly_expired:
                    tracker.expired = started.subject_track_keys
                    analysis.advance(self.batch(10, None))
                else:
                    for second in range(10, 35):
                        analysis.advance(self.batch(second, .5, confidence=.1))
                        self.assertEqual(len(analysis._states), 1)
                    analysis.advance(self.batch(35, .5, confidence=.1))
                self.assertEqual(analysis._states, {})

    def test_default_trackers_keep_camera_state_and_closures_separate(self):
        analysis = ParkingAnalysis(self.policy)
        records = []
        for second in range(4):
            for camera in ("north", "south"):
                records.extend(analysis.advance(self.batch(second, camera=camera)))
        self.assertEqual({record.camera_id for record in records}, {"north", "south"})
        ended = analysis.advance(self.batch(4, camera="north", epoch=2))
        self.assertEqual([record.camera_id for record in ended], ["north"])
        self.assertEqual(len(analysis.finalize("south", 1, self.origin + timedelta(seconds=5))), 1)
        self.assertEqual(analysis.finalize("north", 1, self.origin + timedelta(seconds=5)), [])

    def test_finalize_clamps_time_and_cleans_states_and_default_trackers(self):
        analysis = ParkingAnalysis(self.policy)
        self.start(analysis)
        ended = analysis.finalize("gate", 1, self.origin)
        self.assertEqual(ended[0].ended_at, self.origin + timedelta(seconds=3))
        self.assertEqual(analysis._states, {})
        self.assertEqual(analysis._trackers, {})
        for epoch in range(2, 100):
            analysis.advance(self.batch(epoch, epoch=epoch, sequence=1))
            analysis.finalize("gate", epoch, self.origin + timedelta(seconds=epoch))
        self.assertEqual(analysis._states, {})
        self.assertEqual(analysis._trackers, {})
        self.assertEqual(len(analysis._streams), 1)

    def test_polygon_boundary_is_inside(self):
        self.start(x=.2)

    def test_invalid_policy_and_nonfinite_observations_are_rejected(self):
        for updates in ({"stationary_seconds": 0}, {"stationary_seconds": math.inf},
                        {"exit_seconds": -1}, {"cooldown_seconds": math.nan},
                        {"movement_threshold": -1}, {"movement_threshold": 2},
                        {"max_observation_gap_seconds": 0}, {"minimum_confidence": 1.1},
                        {"target_labels": ()}, {"target_labels": "car"}, {"target_labels": ("car", "car")},
                        {"polygon": ((0, 0), (1, 1), (math.nan, .5))},
                        {"polygon": ((0, 0), (.5, .5), (1, 1))}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.policy, **updates)
        invalid = replace(self.batch(0), width=0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.analysis.advance(invalid)
        invalid = replace(self.batch(0), detections=(Detection("car", .9, BBox(math.nan, 0, 1, 1)),))
        with self.assertRaisesRegex(ValueError, "coordinates"):
            self.analysis.advance(invalid)
        self.assertEqual(self.tracker.calls, 0)
