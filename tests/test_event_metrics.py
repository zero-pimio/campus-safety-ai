import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from campus_safety_ai.contracts import EventRecord, iso_time
from campus_safety_ai.core.event_metrics import evaluate_fall_events

BASE = datetime(2026, 9, 23, tzinfo=UTC)


def at(seconds: float) -> str:
    return iso_time(BASE + timedelta(seconds=seconds))


def interval(start: float, end: float, camera: str = "cam", epoch: int = 1, **extra) -> dict:
    return {"cameraId": camera, "sourceEpoch": epoch, "startedAt": at(start), "endedAt": at(end), **extra}


def truth(*events: tuple[float, float], duration: float = 60) -> dict:
    return {"sessions": [interval(0, duration)], "events": [interval(*event) for event in events]}


def alarm(
    event_id: str, observed: float, *, started: float | None = None,
    camera: str = "cam", epoch: int = 1, phase: str = "START", event_type: str = "person_falling",
) -> dict:
    stamp = BASE + timedelta(seconds=observed)
    record = EventRecord(
        schema_version="1.0", event_id=event_id, revision=1, phase=phase,
        event_type=event_type, severity="warning", edge_id="edge", camera_id=camera,
        started_at=BASE + timedelta(seconds=observed if started is None else started),
        observed_at=stamp, ended_at=stamp if phase == "END" else None,
        subject_track_keys=(), confidence=0.8, model_version="fall", config_version="v1",
        idempotency_key=f"{event_id}:1", status="CLOSED" if phase == "END" else "OPEN",
    )
    return {**record.to_dict(), "sourceEpoch": epoch}


def observation(t: float, score: float | None = 0.2, *, sequence: int | None = None, **extra) -> dict:
    return {
        "cameraId": "cam", "sourceEpoch": 1, "sequence": int(t) if sequence is None else sequence,
        "observedAt": at(t), "windowStartedAt": at(max(0, t - 1)), "windowEndedAt": at(t),
        "score": score, "reason": "warmup" if score is None else None, **extra,
    }


class EventMetricsTests(unittest.TestCase):
    def test_counts_duplicate_alarm_miss_precision_and_delay(self) -> None:
        rows = [alarm("a", 12, started=8), alarm("b", 50), alarm("a", 12, started=8)]
        rows.append(alarm("a", 15, started=8, phase="END"))
        result = evaluate_fall_events(truth((10, 20), (30, 40)), rows, [])
        metric = result["metrics"]
        self.assertEqual((metric["true_positives"], metric["false_positives"], metric["false_negatives"]), (1, 1, 1))
        self.assertEqual((metric["precision"], metric["recall"], metric["miss_rate"]), (0.5, 0.5, 0.5))
        self.assertEqual(metric["false_alarms_per_hour"], 60)
        self.assertEqual(metric["normal_hours"], 40 / 3600)
        self.assertEqual(metric["delay_seconds"]["mean"], 2)
        self.assertEqual(result["matching"]["duplicate_start_count"], 1)

    def test_long_alarm_and_updates_cannot_detect_two_truth_events(self) -> None:
        rows = [alarm("long", 12), alarm("long", 32, started=12, phase="UPDATE")]
        rows.append(alarm("long", 50, started=12, phase="END"))
        report = evaluate_fall_events(truth((10, 20), (30, 40)), rows, [])
        self.assertEqual(report["metrics"]["true_positives"], 1)
        self.assertEqual(report["metrics"]["false_negatives"], 1)

    def test_started_at_is_not_alarm_time_and_tolerance_is_explicit(self) -> None:
        row = alarm("late", 25, started=12)
        report = evaluate_fall_events(truth((10, 20)), [row], [])
        self.assertEqual(report["metrics"]["true_positives"], 0)
        tolerant = evaluate_fall_events(truth((10, 20)), [row], [], tolerance_seconds=5)
        self.assertEqual(tolerant["metrics"]["true_positives"], 1)
        self.assertEqual(tolerant["metrics"]["delay_seconds"]["mean"], 15)

    def test_matching_uses_earliest_deadline_and_at_most_one_alarm_per_truth(self) -> None:
        report = evaluate_fall_events(
            truth((10, 12), (14, 16)),
            [alarm("middle", 13), alarm("late", 17), alarm("duplicate_detection", 18)], [],
            tolerance_seconds=2,
        )
        self.assertEqual(report["metrics"]["true_positives"], 2)
        self.assertEqual(report["metrics"]["false_positives"], 1)
        self.assertEqual([item["truth_event_id"] for item in report["matches"]], ["truth-0", "truth-1"])

    def test_explicit_epochs_and_cameras_prevent_cross_stream_matches(self) -> None:
        annotated = truth((10, 20))
        annotated["sessions"].extend([interval(0, 60, epoch=2), interval(0, 60, camera="other")])
        report = evaluate_fall_events(
            annotated, [alarm("different_epoch", 12, epoch=2), alarm("different_camera", 12, camera="other")], [],
        )
        self.assertEqual(report["metrics"]["true_positives"], 0)
        self.assertEqual(report["metrics"]["false_positives"], 2)
        self.assertEqual(report["metrics"]["annotated_hours"], 180 / 3600)

    def test_tolerance_cannot_match_across_disjoint_sessions(self) -> None:
        annotated = {"sessions": [interval(0, 20), interval(21, 40)], "events": [interval(10, 19)]}
        result = evaluate_fall_events(annotated, [alarm("next_session", 22)], [], tolerance_seconds=10)
        self.assertEqual(result["metrics"]["true_positives"], 0)

    def test_coverage_does_not_fill_gaps_or_extrapolate_edges(self) -> None:
        rows = [observation(2, None), observation(3), observation(4), observation(8), observation(9, None)]
        report = evaluate_fall_events(truth(duration=10), [], rows, max_observation_gap_seconds=2)
        coverage = report["coverage"]
        self.assertEqual((coverage["known_seconds"], coverage["unknown_seconds"], coverage["unobserved_seconds"]), (2, 1, 7))
        self.assertEqual(coverage["unknown_observation_count"], 2)
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["unobserved_fraction"], 0.7)

    def test_absent_or_single_observation_has_zero_time_coverage(self) -> None:
        for rows in ([], [observation(0)], [observation(60)]):
            with self.subTest(rows=rows):
                coverage = evaluate_fall_events(truth(), [], rows)["coverage"]
                self.assertEqual(coverage["known_seconds"], 0)
                self.assertEqual(coverage["unobserved_seconds"], 60)

    def test_complete_unknown_coverage_includes_warmup(self) -> None:
        rows = [observation(0, None), observation(1, None), observation(2, None)]
        report = evaluate_fall_events(truth(duration=2), [], rows)
        self.assertEqual(report["coverage"]["unknown_fraction"], 1)
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["known_fraction"], 0)

    def test_empty_ground_truth_is_valid_but_rates_with_no_denominator_are_null(self) -> None:
        report = evaluate_fall_events(truth(), [], [])
        self.assertIsNone(report["metrics"]["recall"])
        self.assertIsNone(report["metrics"]["miss_rate"])
        self.assertIsNone(report["metrics"]["precision"])
        self.assertEqual(report["metrics"]["false_alarms_per_hour"], 0)
        self.assertEqual(report["metrics"]["normal_hours"], 60 / 3600)

    def test_orphan_revision_is_reported_without_inventing_start_or_coverage(self) -> None:
        report = evaluate_fall_events(truth((10, 20)), [alarm("orphan", 12, phase="END")], [])
        self.assertEqual(report["matching"]["orphan_event_ids"], ["orphan"])
        self.assertEqual(report["metrics"]["alarm_count"], 0)
        self.assertEqual(report["coverage"]["unobserved_fraction"], 1)
        self.assertTrue(any("event input is incomplete" in message for message in report["warnings"]))

    def test_rejects_missing_epoch_conflicting_start_and_out_of_session_alarm(self) -> None:
        missing = alarm("x", 12)
        del missing["sourceEpoch"]
        for rows in ([missing], [alarm("x", 12), alarm("x", 13)], [alarm("x", 61)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_fall_events(truth(), rows, [])

    def test_rejects_invalid_truth_durations_overlaps_and_uncontained_event(self) -> None:
        invalid = [
            {"sessions": [], "events": []},
            {"sessions": [interval(0, 0)], "events": []},
            {"sessions": [interval(0, 20), interval(10, 30)], "events": []},
            truth((10, 30), (20, 40)), truth((50, 70)), truth((10, 10)),
        ]
        for value in invalid:
            with self.subTest(truth=value), self.assertRaises(ValueError):
                evaluate_fall_events(value, [], [])

    def test_rejects_invalid_observations(self) -> None:
        invalid = [
            observation(1, float("nan")), observation(1, True), observation(1, 1.1),
            observation(1, None, reason=""), observation(1, sourceEpoch=True),
            observation(1, windowEndedAt=at(2)), observation(1, windowStartedAt=at(2)),
            observation(1, observedAt="2026-09-23T00:00:01"),
            observation(1, sequence="1"), observation(61),
        ]
        missing = deepcopy(observation(1))
        del missing["score"]
        invalid.append(missing)
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate_fall_events(truth(), [], [row])
        for rows in ([observation(1), observation(1)], [observation(1, sequence=3), observation(2, sequence=2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_fall_events(truth(), [], rows)

    def test_rejects_invalid_options(self) -> None:
        for options in (
            {"tolerance_seconds": -1}, {"tolerance_seconds": float("inf")},
            {"max_observation_gap_seconds": 0}, {"max_observation_gap_seconds": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_fall_events(truth(), [], [], **options)


if __name__ == "__main__":
    unittest.main()
