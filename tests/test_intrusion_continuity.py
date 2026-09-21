from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from campus_safety_ai.contracts import BBox, Detection, Detections
from campus_safety_ai.core.event_analysis import EventAnalysis, IntrusionPolicy, SimpleIoUTracker, _inside_polygon


ORIGIN = datetime(2026, 9, 21, tzinfo=UTC)
POLYGON = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
INSIDE = BBox(550, 100, 750, 800)


def batch(second: float, sequence: int, confidence: float | None = 0.9, epoch: int = 1) -> Detections:
    return Detections(
        camera_id="gate", source_epoch=epoch, sequence=sequence,
        captured_at=ORIGIN + timedelta(seconds=second), width=1000, height=1000,
        detections=() if confidence is None else (Detection("person", confidence, INSIDE),),
        model_version="test", inference_ms=1,
    )


def analysis(**kwargs) -> EventAnalysis:
    return EventAnalysis(
        IntrusionPolicy(edge_id="edge", zone_id="restricted", polygon=POLYGON, enter_seconds=2, **kwargs),
        tracker=SimpleIoUTracker(max_gap_seconds=60),
    )


class IntrusionContinuityTests(unittest.TestCase):
    def test_missing_detection_breaks_pending_confirmation(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        detector.advance(batch(1, 2, confidence=None))
        self.assertEqual(detector.advance(batch(2, 3)), [])
        self.assertEqual([record.phase for record in detector.advance(batch(4, 4))], ["START"])

    def test_low_confidence_breaks_pending_confirmation(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        detector.advance(batch(1, 2, confidence=0.1))
        self.assertEqual(detector.advance(batch(2, 3)), [])

    def test_long_observation_gap_restarts_confirmation_even_if_tracker_retains_id(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        self.assertEqual(detector.advance(batch(20, 2)), [])

    def test_timestamp_regression_is_rejected_without_consuming_sequence(self) -> None:
        detector = analysis()
        detector.advance(batch(1, 1))
        with self.assertRaises(ValueError):
            detector.advance(batch(0, 2))
        self.assertEqual([record.phase for record in detector.advance(batch(3, 2))], ["START"])

    def test_all_polygon_edges_and_vertices_are_inside(self) -> None:
        for point in ((0.5, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.5), (1, 0.5), (0.75, 0), (0.75, 1)):
            with self.subTest(point=point):
                self.assertTrue(_inside_polygon(point, POLYGON))

    def test_finalize_clears_sequence_bookkeeping(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        detector.finalize("gate", 1, ORIGIN)
        self.assertEqual(detector._last_sequence, {})
        self.assertEqual(detector._last_captured_at, {})
        self.assertEqual(detector.tracker._objects, {})
        self.assertIsNone(detector._active_stream)

    def test_lower_confidence_observations_end_open_event_after_gap(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        started = detector.advance(batch(2, 2))[0]
        self.assertEqual(detector.advance(batch(3, 3, confidence=0.1)), [])
        ended = detector.advance(batch(5, 4, confidence=0.1))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].event_id, started.event_id)
        self.assertEqual(ended[0].confidence, 0.0)
        self.assertEqual(detector.advance(batch(6, 5)), [])
        self.assertEqual(detector.advance(batch(8, 6)), [])  # still in cooldown

    def test_missing_detection_breaks_departure_confirmation(self) -> None:
        detector = analysis(exit_seconds=1)
        detector.advance(batch(0, 1))
        detector.advance(batch(2, 2))
        outside = Detection("person", 0.9, BBox(295, 100, 695, 800))
        detector.advance(replace(batch(3, 3), detections=(outside,)))
        detector.advance(batch(3.5, 4, confidence=None))
        self.assertEqual(detector.advance(replace(batch(4, 5), detections=(outside,))), [])
        self.assertEqual(
            [record.phase for record in detector.advance(replace(batch(5, 6), detections=(outside,)))], ["END"]
        )

    def test_new_epoch_closes_old_event_and_drops_old_history(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        started = detector.advance(batch(2, 2))[0]
        ended = detector.advance(batch(3, 1, epoch=2))
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].event_id, started.event_id)
        self.assertEqual(set(detector._last_sequence), {("gate", 2)})
        self.assertEqual(detector.advance(batch(5, 100, epoch=1)), [])
        self.assertEqual([record.phase for record in detector.advance(batch(5, 2, epoch=2))], ["START"])

    def test_camera_switch_ends_event_under_original_camera(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        started = detector.advance(batch(2, 2))[0]
        records = detector.advance(replace(batch(3, 1), camera_id="different-camera"))
        self.assertEqual([record.phase for record in records], ["END"])
        self.assertEqual(records[0].camera_id, "gate")
        self.assertEqual(records[0].event_id, started.event_id)

    def test_epoch_restart_may_reset_clock_without_backdating_old_end(self) -> None:
        detector = analysis()
        detector.advance(batch(10, 1))
        detector.advance(batch(12, 2))
        ended = detector.advance(batch(0, 1, epoch=2))
        self.assertEqual(ended[0].ended_at, ORIGIN + timedelta(seconds=12))

    def test_finalize_refuses_backdated_end_and_preserves_open_event(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        started = detector.advance(batch(2, 2))[0]
        with self.assertRaises(ValueError):
            detector.finalize("gate", 1, ORIGIN + timedelta(seconds=1))
        ended = detector.finalize("gate", 1, ORIGIN + timedelta(seconds=2))
        self.assertEqual(ended[0].event_id, started.event_id)
        self.assertEqual(detector.finalize("gate", 1, ORIGIN + timedelta(seconds=2)), [])
        self.assertEqual(detector.advance(batch(3, 3)), [])  # finalized epoch remains closed
        self.assertEqual(detector.advance(batch(3, 1, epoch=2)), [])

    def test_reconnect_history_is_bounded_by_camera_count(self) -> None:
        detector = analysis()
        for epoch in range(1, 101):
            detector.advance(batch(epoch, 1, epoch=epoch))
        self.assertEqual(len(detector._source_epochs), 1)
        self.assertEqual(len(detector._last_sequence), 1)
        self.assertEqual(len(detector._last_captured_at), 1)
        self.assertEqual(len(detector._states), 1)
        detector.finalize("gate", 100, ORIGIN + timedelta(seconds=100))
        self.assertEqual(detector._states, {})

    def test_max_gap_can_be_configured_without_changing_tracker(self) -> None:
        detector = analysis(max_observation_gap_seconds=0.5)
        detector.advance(batch(0, 1))
        self.assertEqual(detector.advance(batch(2, 2)), [])

    def test_duplicate_does_not_disrupt_valid_confirmation(self) -> None:
        detector = analysis()
        detector.advance(batch(0, 1))
        self.assertEqual(detector.advance(batch(30, 1)), [])
        self.assertEqual([record.phase for record in detector.advance(batch(2, 2))], ["START"])


class IntrusionValidationTests(unittest.TestCase):
    def test_tracker_receives_only_target_class_including_low_confidence_targets(self) -> None:
        class RecordingTracker(SimpleIoUTracker):
            def update(self, observations):
                self.received = observations
                return super().update(observations)

        tracker = RecordingTracker()
        detector = EventAnalysis(IntrusionPolicy("edge", "zone", POLYGON), tracker=tracker)
        observations = replace(batch(0, 1), detections=(
            Detection("person", 0.1, INSIDE), Detection("car", 0.99, INSIDE),
        ))
        self.assertEqual(detector.advance(observations), [])
        self.assertEqual(tracker.received.detections, (observations.detections[0],))
        self.assertEqual(tracker.received.camera_id, observations.camera_id)
        self.assertEqual(tracker.received.captured_at, observations.captured_at)
        self.assertEqual(tracker.received.sequence, observations.sequence)

    def test_non_target_cannot_continue_person_confirmation_in_class_agnostic_tracker(self) -> None:
        class ClassAgnosticTracker(SimpleIoUTracker):
            def update(self, observations):
                # Simulate a tracker that associates boxes without a class gate.
                return super().update(replace(observations, detections=tuple(
                    replace(detection, label="person") for detection in observations.detections
                )))

        detector = EventAnalysis(IntrusionPolicy("edge", "zone", POLYGON), tracker=ClassAgnosticTracker())
        detector.advance(batch(0, 1))
        vehicle = replace(batch(1, 2), detections=(Detection("car", 0.99, INSIDE),))
        self.assertEqual(detector.advance(vehicle), [])
        self.assertEqual(detector.advance(batch(2, 3)), [])
        self.assertEqual([record.phase for record in detector.advance(batch(4, 4))], ["START"])

    def test_policy_rejects_nonfinite_negative_or_invalid_parameters(self) -> None:
        for name in ("enter_seconds", "exit_seconds", "cooldown_seconds", "max_observation_gap_seconds"):
            for value in (-1, float("nan"), float("inf"), True, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    IntrusionPolicy("edge", "zone", POLYGON, **{name: value})
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=value), self.assertRaises(ValueError):
                IntrusionPolicy("edge", "zone", POLYGON, minimum_confidence=value)
        for polygon in (
            ((0, 0), (1, 1)), ((0, 0), (0.5, 0.5), (1, 1)),
            ((0, 0), (1, 0), (float("nan"), 1)), ((0, 0), (1, 0), (1, float("inf"))),
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                IntrusionPolicy("edge", "zone", polygon)

    def test_malformed_batch_fails_before_consuming_sequence(self) -> None:
        detector = analysis()
        for change in (
            {"width": 0}, {"height": -1}, {"sequence": -1}, {"source_epoch": -1},
            {"captured_at": ORIGIN.replace(tzinfo=None)},
            {"detections": (Detection("person", 0.9, BBox(0, 0, float("nan"), 1)),)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                detector.advance(replace(batch(0, 1), **change))
        self.assertEqual(detector._last_sequence, {})

    def test_polygon_boundary_orientation_and_outside(self) -> None:
        for polygon in (POLYGON, tuple(reversed(POLYGON))):
            self.assertTrue(_inside_polygon((1, 1), polygon))
            self.assertFalse(_inside_polygon((1.000001, 1), polygon))
            self.assertFalse(_inside_polygon((float("nan"), 0.5), polygon))

    def test_zero_iou_threshold_accepts_first_detection(self) -> None:
        tracker = SimpleIoUTracker(iou_threshold=0)
        self.assertEqual(len(tracker.update(batch(0, 1)).tracks), 1)

    def test_tracker_rejects_invalid_parameters(self) -> None:
        for values in ({"iou_threshold": 1.1}, {"iou_threshold": float("nan")},
                       {"max_gap_seconds": -1}, {"max_gap_seconds": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SimpleIoUTracker(**values)


if __name__ == "__main__":
    unittest.main()
