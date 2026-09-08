import unittest
from datetime import datetime, timedelta, UTC

from campus_safety_ai.contracts import EventRecord


def record(phase: str = "START", ended_at: datetime | None = None) -> EventRecord:
    started = datetime(2026, 8, 25, 8, 0, tzinfo=UTC)
    return EventRecord(
        schema_version="1.0",
        event_id="evt-1",
        revision=1,
        phase=phase,
        event_type="person_fighting",
        severity="critical",
        edge_id="edge-01",
        camera_id="gate-02",
        started_at=started,
        observed_at=started + timedelta(seconds=5),
        ended_at=ended_at,
        subject_track_keys=("gate-02:1:1:7",),
        confidence=0.87,
        model_version="fight-v1",
        config_version="fight-v1",
        idempotency_key="evt-1:1",
        status="OPEN" if phase != "END" else "CLOSED",
        evidence_uris=("runtime/evidence/snap.jpg",),
    )


class EventRecordContractTests(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        ended = record("END", datetime(2026, 8, 25, 8, 0, 5, tzinfo=UTC))
        self.assertEqual(EventRecord.from_dict(ended.to_dict()), ended)
        open_record = record()
        self.assertEqual(EventRecord.from_dict(open_record.to_dict()), open_record)

    def test_rejects_revision_below_one(self) -> None:
        value = record()
        object.__setattr__(value, "revision", 0)
        with self.assertRaises(ValueError):
            EventRecord.__post_init__(value)

    def test_rejects_confidence_outside_probability_range(self) -> None:
        invalid = record()
        object.__setattr__(invalid, "confidence", 1.5)
        with self.assertRaises(ValueError):
            EventRecord.__post_init__(invalid)

    def test_rejects_start_record_carrying_ended_at(self) -> None:
        with self.assertRaises(ValueError):
            record("START", datetime(2026, 8, 25, 8, 0, 5, tzinfo=UTC))

    def test_rejects_end_record_missing_ended_at(self) -> None:
        with self.assertRaises(ValueError):
            record("END", None)

    def test_rejects_naive_timestamps(self) -> None:
        value = record()
        object.__setattr__(value, "observed_at", datetime(2026, 8, 25, 8, 0))
        with self.assertRaises(ValueError):
            EventRecord.__post_init__(value)

    def test_rejects_unknown_phase_and_empty_identity(self) -> None:
        unknown_phase = record()
        object.__setattr__(unknown_phase, "phase", "PAUSE")
        with self.assertRaises(ValueError):
            EventRecord.__post_init__(unknown_phase)

        empty_key = record()
        object.__setattr__(empty_key, "idempotency_key", "")
        with self.assertRaises(ValueError):
            EventRecord.__post_init__(empty_key)


if __name__ == "__main__":
    unittest.main()
