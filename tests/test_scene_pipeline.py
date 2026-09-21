import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from campus_safety_ai.contracts import BBox, Detection, Detections, FramePacket
from campus_safety_ai.core.event_analysis import EventAnalysis, IntrusionPolicy
from campus_safety_ai.core.parking_analysis import ParkingAnalysis, ParkingPolicy
from campus_safety_ai.core.scene_pipeline import SceneVideoPipeline


class FixedPerception:
    def __init__(self, label, *, fail_sequence=None, failure=None, changed_metadata=None):
        self.label = label
        self.fail_sequence = fail_sequence
        self.failure = failure
        self.changed_metadata = changed_metadata or {}
        self.calls = []

    def detect(self, frame):
        self.calls.append(frame.sequence)
        if frame.sequence == self.fail_sequence:
            raise self.failure
        batch = Detections(
            frame.camera_id, frame.source_epoch, frame.sequence, frame.captured_at,
            frame.width, frame.height, (Detection(self.label, .9, BBox(30, 20, 50, 70)),),
            "scene-test", 1,
        )
        return replace(batch, **self.changed_metadata) if frame.sequence == 2 else batch


class ScenePipelineTests(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2026, 9, 21, tzinfo=UTC)

    def frames(self, count=3):
        for index in range(count):
            observed = self.origin + timedelta(seconds=index)
            yield FramePacket("gate", 1, index + 1, observed, observed, 100, 100, image=object())

    @staticmethod
    def analysis(label, *, instant=False):
        polygon = ((0, 0), (1, 0), (1, 1), (0, 1))
        if label == "person":
            return EventAnalysis(IntrusionPolicy(
                "edge", "zone", polygon, enter_seconds=0 if instant else 1,
            ))
        return ParkingAnalysis(ParkingPolicy("edge", "zone", polygon, stationary_seconds=1))

    def test_both_rules_emit_per_frame_and_close_same_event_at_eof(self):
        for label, event_type in (("person", "intrusion"), ("car", "illegal_parking")):
            with self.subTest(label=label):
                perception = FixedPerception(label)
                stream = SceneVideoPipeline(perception, self.analysis(label)).stream(self.frames())
                self.assertEqual(perception.calls, [])
                first = next(stream)
                self.assertEqual(first.events, ())
                self.assertEqual(perception.calls, [1])
                second = next(stream)
                self.assertEqual([record.phase for record in second.events], ["START"])
                start = second.events[0]
                self.assertEqual(start.event_type, event_type)
                self.assertEqual(start.started_at, self.origin)
                self.assertEqual(second.frame.sequence, 2)
                self.assertEqual(second.detections.sequence, 2)
                self.assertIsNone(second.finish_reason)
                third = next(stream)
                self.assertEqual(third.events, ())
                closed = next(stream)
                self.assertEqual(closed.finish_reason, "eof")
                self.assertIsNone(closed.frame)
                self.assertIsNone(closed.detections)
                self.assertEqual([record.phase for record in closed.events], ["END"])
                end = closed.events[0]
                self.assertEqual(end.event_id, start.event_id)
                self.assertEqual(end.ended_at, third.frame.captured_at)
                self.assertEqual(end.revision, start.revision + 1)
                self.assertEqual(end.status, "CLOSED")
                with self.assertRaises(StopIteration):
                    next(stream)

    def test_decoder_failure_emits_closure_before_reraising_original_error(self):
        failure = OSError("decoder stopped unexpectedly")

        def frames():
            yield from self.frames(2)
            raise failure

        for label in ("person", "car"):
            with self.subTest(label=label):
                stream = SceneVideoPipeline(FixedPerception(label), self.analysis(label)).stream(frames())
                next(stream)
                started = next(stream).events[0]
                closed = next(stream)
                self.assertEqual(closed.finish_reason, "error")
                self.assertEqual([record.phase for record in closed.events], ["END"])
                self.assertEqual(closed.events[0].event_id, started.event_id)
                self.assertEqual(closed.events[0].ended_at, self.origin + timedelta(seconds=1))
                with self.assertRaises(OSError) as caught:
                    next(stream)
                self.assertIs(caught.exception, failure)

    def test_perception_failure_closes_at_last_accepted_frame(self):
        failure = RuntimeError("inference unavailable")
        perception = FixedPerception("car", fail_sequence=3, failure=failure)
        stream = SceneVideoPipeline(perception, self.analysis("car")).stream(self.frames())
        next(stream)
        started = next(stream).events[0]
        closed = next(stream)
        self.assertEqual(perception.calls, [1, 2, 3])
        self.assertEqual(closed.finish_reason, "error")
        self.assertEqual(closed.events[0].event_id, started.event_id)
        self.assertEqual(closed.events[0].ended_at, self.origin + timedelta(seconds=1))
        with self.assertRaises(RuntimeError) as caught:
            next(stream)
        self.assertIs(caught.exception, failure)

    def test_changed_batch_metadata_is_rejected_and_prior_event_closed(self):
        for changes in (
            {"camera_id": "other"}, {"source_epoch": 2}, {"sequence": 8},
            {"captured_at": self.origin + timedelta(seconds=20)}, {"width": 200}, {"height": 200},
        ):
            with self.subTest(changes=changes):
                perception = FixedPerception("person", changed_metadata=changes)
                stream = SceneVideoPipeline(perception, self.analysis("person", instant=True)).stream(
                    self.frames(2)
                )
                started = next(stream).events[0]
                closed = next(stream)
                self.assertEqual(closed.finish_reason, "error")
                self.assertIsNone(closed.frame)
                self.assertEqual([record.phase for record in closed.events], ["END"])
                self.assertEqual(closed.events[0].event_id, started.event_id)
                self.assertEqual(closed.events[0].ended_at, self.origin)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    next(stream)

    def test_failure_before_first_observation_has_no_synthetic_closure(self):
        failure = RuntimeError("first inference failed")
        perception = FixedPerception("person", fail_sequence=1, failure=failure)
        stream = SceneVideoPipeline(perception, self.analysis("person")).stream(self.frames())
        with self.assertRaises(RuntimeError) as caught:
            next(stream)
        self.assertIs(caught.exception, failure)

    def test_empty_input_has_no_event_or_closure(self):
        perception = FixedPerception("person")
        self.assertEqual(list(SceneVideoPipeline(perception, self.analysis("person")).stream(())), [])
        self.assertEqual(perception.calls, [])


if __name__ == "__main__":
    unittest.main()
