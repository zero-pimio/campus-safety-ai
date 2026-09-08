import json
import tempfile
import unittest
from datetime import datetime, timedelta, UTC
from pathlib import Path

from campus_safety_ai.contracts import BBox, Detection, Detections
from campus_safety_ai.core.event_analysis import EventAnalysis, IntrusionPolicy, SimpleIoUTracker


def batch(
    sequence: int,
    captured_at: datetime,
    detections: tuple[Detection, ...],
    camera_id: str = "gate-02",
    source_epoch: int = 1,
) -> Detections:
    return Detections(
        camera_id=camera_id,
        source_epoch=source_epoch,
        sequence=sequence,
        captured_at=captured_at,
        width=1000,
        height=1000,
        detections=detections,
        model_version="fake-v1",
        inference_ms=1,
    )


class IntrusionFinalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2026, 8, 25, tzinfo=UTC)
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

    def test_finalize_closes_event_stuck_in_open_state(self) -> None:
        inside = Detection("person", 0.9, BBox(550, 100, 750, 800))
        self.analysis.advance(batch(1, self.origin, (inside,)))
        started = self.analysis.advance(batch(2, self.origin + timedelta(seconds=2), (inside,)))
        self.assertEqual([record.phase for record in started], ["START"])

        ended_at = self.origin + timedelta(seconds=30)
        finalized = self.analysis.finalize("gate-02", 1, ended_at)

        self.assertEqual([record.phase for record in finalized], ["END"])
        self.assertEqual(finalized[0].event_id, started[0].event_id)
        self.assertNotEqual(finalized[0].idempotency_key, started[0].idempotency_key)
        self.assertGreater(finalized[0].revision, started[0].revision)

    def test_finalize_only_touches_requested_epoch(self) -> None:
        other = Detection("person", 0.9, BBox(550, 100, 750, 800))
        self.analysis.advance(batch(1, self.origin, (other,), camera_id="gate-03"))
        self.analysis.advance(batch(2, self.origin + timedelta(seconds=2), (other,), camera_id="gate-03"))
        self.assertEqual(self.analysis.finalize("gate-02", 1, self.origin), [])

    def test_finalize_discards_states_so_services_do_not_leak(self) -> None:
        inside = Detection("person", 0.9, BBox(550, 100, 750, 800))
        self.analysis.advance(batch(1, self.origin, (inside,)))
        self.analysis.advance(batch(2, self.origin + timedelta(seconds=2), (inside,)))

        self.analysis.finalize("gate-02", 1, self.origin + timedelta(seconds=30))

        self.assertEqual(self.analysis._states, {})

    def test_source_epoch_change_closes_open_event(self) -> None:
        inside = Detection("person", 0.9, BBox(550, 100, 750, 800))
        self.analysis.advance(batch(1, self.origin, (inside,)))
        started = self.analysis.advance(batch(2, self.origin + timedelta(seconds=2), (inside,)))
        self.assertEqual([record.phase for record in started], ["START"])

        # Camera reconnect: same camera, new source epoch, first frame of the new session.
        ended = self.analysis.advance(
            batch(1, self.origin + timedelta(seconds=30), (inside,), source_epoch=2)
        )

        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].event_id, started[0].event_id)
        self.assertFalse(any(key.startswith("gate-02:1:") for key in self.analysis._states))


class TrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2026, 8, 25, tzinfo=UTC)
        self.tracker = SimpleIoUTracker(iou_threshold=0.25, max_gap_seconds=2.0)

    def test_same_object_keeps_one_track_key(self) -> None:
        first = batch(1, self.origin, (Detection("person", 0.9, BBox(100, 100, 200, 300)),))
        second = batch(2, self.origin + timedelta(seconds=0.5), (Detection("person", 0.8, BBox(105, 100, 205, 300)),))

        first_tracks = self.tracker.update(first).tracks
        second_tracks = self.tracker.update(second).tracks

        self.assertEqual(len(first_tracks), len(second_tracks), 1)
        self.assertEqual(first_tracks[0].track_key, second_tracks[0].track_key)

    def test_long_gap_assigns_new_track_key(self) -> None:
        first = batch(1, self.origin, (Detection("person", 0.9, BBox(100, 100, 200, 300)),))
        late = batch(2, self.origin + timedelta(seconds=10), (Detection("person", 0.8, BBox(100, 100, 200, 300)),))

        first_tracks = self.tracker.update(first).tracks
        late_tracks = self.tracker.update(late).tracks

        self.assertNotEqual(first_tracks[0].track_key, late_tracks[0].track_key)

    def test_new_source_epoch_resets_ids(self) -> None:
        epoch_one = Detections(
            camera_id="gate-02",
            source_epoch=1,
            sequence=1,
            captured_at=self.origin,
            width=1000,
            height=1000,
            detections=(Detection("person", 0.9, BBox(100, 100, 200, 300)),),
            model_version="fake-v1",
            inference_ms=1,
        )
        epoch_two = Detections(
            camera_id="gate-02",
            source_epoch=2,
            sequence=1,
            captured_at=self.origin + timedelta(seconds=1),
            width=1000,
            height=1000,
            detections=(Detection("person", 0.9, BBox(100, 100, 200, 300)),),
            model_version="fake-v1",
            inference_ms=1,
        )
        one = self.tracker.update(epoch_one).tracks
        two = self.tracker.update(epoch_two).tracks
        self.assertNotEqual(one[0].track_key, two[0].track_key)

    def test_expired_track_is_reported(self) -> None:
        first = batch(1, self.origin, (Detection("person", 0.9, BBox(100, 100, 200, 300)),))
        key = self.tracker.update(first).tracks[0].track_key

        expired = self.tracker.update(batch(2, self.origin + timedelta(seconds=3), ()))

        self.assertEqual(expired.expired_track_keys, (key,))

    def test_source_epoch_switch_reports_old_tracks_as_expired(self) -> None:
        first = batch(1, self.origin, (Detection("person", 0.9, BBox(100, 100, 200, 300)),))
        key = self.tracker.update(first).tracks[0].track_key

        expired = self.tracker.update(
            batch(1, self.origin + timedelta(seconds=1), (), source_epoch=2)
        )

        self.assertEqual(expired.expired_track_keys, (key,))


class LostTrackClosureTests(unittest.TestCase):
    def test_open_intrusion_closes_when_track_expires(self) -> None:
        origin = datetime(2026, 8, 25, tzinfo=UTC)
        analysis = EventAnalysis(
            IntrusionPolicy(
                edge_id="edge-01",
                zone_id="all",
                polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
                enter_seconds=0,
                exit_seconds=1,
                cooldown_seconds=0,
            ),
            tracker=SimpleIoUTracker(max_gap_seconds=2),
        )
        inside = Detection("person", 0.9, BBox(100, 100, 200, 300))
        started = analysis.advance(batch(1, origin, (inside,)))

        ended = analysis.advance(batch(2, origin + timedelta(seconds=3), ()))

        self.assertEqual([record.phase for record in started], ["START"])
        self.assertEqual([record.phase for record in ended], ["END"])
        self.assertEqual(ended[0].confidence, 0.0)


class DeliveryContextTests(unittest.TestCase):
    def test_context_manager_flushes_pending_on_exit(self) -> None:
        from campus_safety_ai.contracts import EventRecord
        from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination

        now = datetime(2026, 8, 25, tzinfo=UTC)
        record = EventRecord(
            schema_version="1.0",
            event_id="event-1",
            revision=1,
            phase="START",
            event_type="intrusion",
            severity="warning",
            edge_id="edge-01",
            camera_id="gate-02",
            started_at=now,
            observed_at=now,
            ended_at=None,
            subject_track_keys=("t",),
            confidence=0.9,
            model_version="fake-v1",
            config_version="intrusion-v1",
            idempotency_key="event-1:1",
            status="OPEN",
        )

        with tempfile.TemporaryDirectory() as directory:
            target = InMemoryDestination()
            delivery = EventDelivery(Path(directory) / "outbox.sqlite3", target)
            with delivery as entered:
                # Enqueue without flushing so __exit__ has work to do.
                with entered.connection:
                    entered.connection.execute(
                        "INSERT OR IGNORE INTO outbox(idempotency_key, payload) VALUES (?, ?)",
                        (
                            record.idempotency_key,
                            json.dumps(record.to_dict(), ensure_ascii=False),
                        ),
                    )
                self.assertEqual(len(target.records), 0)
            self.assertEqual(len(target.records), 1)


if __name__ == "__main__":
    unittest.main()
