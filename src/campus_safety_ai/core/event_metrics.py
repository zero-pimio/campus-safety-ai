"""Event-level fall evaluation against explicitly annotated camera sessions.

Classification windows are not ground-truth events. An alarm's actual START
``observedAt`` is matched once, within its camera, epoch and annotated session.
Coverage is measured separately so missing inference never becomes normal time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from statistics import mean, median
from typing import Any

from campus_safety_ai.contracts import EventRecord, iso_time, parse_time


@dataclass(frozen=True)
class _Interval:
    camera_id: str
    source_epoch: int
    start: datetime
    end: datetime
    label: str


@dataclass(frozen=True)
class _Observation:
    sequence: int
    observed_at: datetime
    score: float | None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def _time(value: Any, name: str) -> datetime:
    try:
        return parse_time(_text(value, name))
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"missing required field: {key}")
    return value[key]


def _interval(value: Any, label: str) -> _Interval:
    value = _object(value, label)
    interval = _Interval(
        _text(_required(value, "cameraId"), "cameraId"),
        _integer(_required(value, "sourceEpoch"), "sourceEpoch"),
        _time(_required(value, "startedAt"), "startedAt"),
        _time(_required(value, "endedAt"), "endedAt"),
        label,
    )
    if interval.end <= interval.start:
        raise ValueError(f"{label} must have positive duration")
    return interval


def _session_for(
    sessions: Sequence[_Interval], camera: str, epoch: int, start: datetime, end: datetime,
) -> int:
    matches = [
        index for index, session in enumerate(sessions)
        if (session.camera_id, session.source_epoch) == (camera, epoch)
        and session.start <= start <= end <= session.end
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{camera}/{epoch} at {iso_time(start)} must belong to exactly one annotated session; "
            f"found {len(matches)} (shared session endpoints are ambiguous)"
        )
    return matches[0]


def _non_overlapping(intervals: Sequence[_Interval], name: str) -> None:
    grouped: dict[tuple[str, int], list[_Interval]] = defaultdict(list)
    for interval in intervals:
        grouped[(interval.camera_id, interval.source_epoch)].append(interval)
    for group in grouped.values():
        ordered = sorted(group, key=lambda interval: interval.start)
        for previous, current in pairwise(ordered):
            if current.start < previous.end:
                raise ValueError(f"{name} must not overlap within a camera/sourceEpoch")


def evaluate_fall_events(
    ground_truth: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float = 0,
    max_observation_gap_seconds: float = 2,
    event_type: str = "person_falling",
) -> dict[str, Any]:
    """Return event counts and coverage without extrapolating missing observations.

    Every event JSON row requires an explicit ``sourceEpoch`` alongside the
    EventRecord fields. START rows are deduplicated by eventId; conflicting START
    rows are rejected. Non-START revisions never count as additional alarms.
    Unknown observation scores must be null with a non-empty ``reason``.

    Between consecutive observations, the earlier row's known/unknown state owns
    the interval only when the gap is <= max_observation_gap_seconds. Longer
    gaps, and the session's leading/trailing gaps, are entirely unobserved.
    """
    tolerance = _number(tolerance_seconds, "tolerance_seconds")
    max_gap = _number(max_observation_gap_seconds, "max_observation_gap_seconds", positive=True)
    _text(event_type, "event_type")
    ground_truth = _object(ground_truth, "ground_truth")
    for name in ("sessions", "events"):
        if not isinstance(_required(ground_truth, name), list):
            raise ValueError(f"ground_truth.{name} must be an array")
    sessions = [_interval(item, f"session-{index}") for index, item in enumerate(ground_truth["sessions"])]
    if not sessions:
        raise ValueError("ground_truth.sessions must contain positive annotated duration")
    _non_overlapping(sessions, "sessions")
    truths: list[_Interval] = []
    truth_session: list[int] = []
    truth_ids: set[str] = set()
    for index, item in enumerate(ground_truth["events"]):
        item = _object(item, f"truth-{index}")
        label = _text(item["eventId"], "eventId") if "eventId" in item else f"truth-{index}"
        if label in truth_ids:
            raise ValueError(f"duplicate ground-truth eventId: {label}")
        truth_ids.add(label)
        interval = _interval(item, label)
        truth_session.append(_session_for(
            sessions, interval.camera_id, interval.source_epoch, interval.start, interval.end,
        ))
        truths.append(interval)
    _non_overlapping(truths, "ground-truth events")

    starts: dict[str, tuple[EventRecord, int, int]] = {}
    other_ids: set[str] = set()
    duplicate_starts = 0
    ignored_other_type_records = 0
    for raw in events:
        raw = _object(raw, "event")
        epoch = _integer(_required(raw, "sourceEpoch"), "sourceEpoch")
        _text(_required(raw, "cameraId"), "cameraId")
        _text(_required(raw, "eventId"), "eventId")
        revision = _integer(_required(raw, "revision"), "revision")
        if revision < 1:
            raise ValueError("event revision must start at 1")
        confidence = _number(_required(raw, "confidence"), "confidence")
        if confidence > 1:
            raise ValueError("event confidence must be between 0 and 1")
        try:
            event = EventRecord.from_dict(dict(raw))
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"invalid EventRecord: {error}") from error
        if event.observed_at < event.started_at:
            raise ValueError("event observedAt cannot precede startedAt")
        if event.ended_at is not None and not event.started_at <= event.ended_at <= event.observed_at:
            raise ValueError("event endedAt must lie between startedAt and observedAt")
        if event.event_type != event_type:
            ignored_other_type_records += 1
            continue
        if event.phase != "START":
            other_ids.add(event.event_id)
            continue
        session_index = _session_for(
            sessions, event.camera_id, epoch, event.observed_at, event.observed_at,
        )
        candidate = (event, epoch, session_index)
        if event.event_id in starts:
            if candidate != starts[event.event_id]:
                raise ValueError(f"conflicting START records for eventId: {event.event_id}")
            duplicate_starts += 1
        else:
            starts[event.event_id] = candidate

    by_session: dict[int, list[_Observation]] = defaultdict(list)
    for raw in observations:
        raw = _object(raw, "observation")
        camera = _text(_required(raw, "cameraId"), "cameraId")
        epoch = _integer(_required(raw, "sourceEpoch"), "sourceEpoch")
        sequence = _integer(_required(raw, "sequence"), "sequence")
        observed_at = _time(_required(raw, "observedAt"), "observedAt")
        window_start = _time(_required(raw, "windowStartedAt"), "windowStartedAt")
        window_end = _time(_required(raw, "windowEndedAt"), "windowEndedAt")
        if not window_start <= window_end <= observed_at:
            raise ValueError("observation must satisfy windowStartedAt <= windowEndedAt <= observedAt")
        score = _required(raw, "score")
        if score is None:
            _text(_required(raw, "reason"), "unknown observation reason")
        else:
            score = _number(score, "score")
            if score > 1:
                raise ValueError("observation score must be between 0 and 1")
        session_index = _session_for(sessions, camera, epoch, observed_at, observed_at)
        by_session[session_index].append(_Observation(sequence, observed_at, score))

    coverage_sessions: list[dict[str, Any]] = []
    total_known = total_unknown = 0.0
    known_count = unknown_count = 0
    for index, session in enumerate(sessions):
        rows = sorted(by_session[index], key=lambda row: row.observed_at)
        known = unknown = 0.0
        known_count += sum(row.score is not None for row in rows)
        unknown_count += sum(row.score is None for row in rows)
        for previous, current in pairwise(rows):
            if current.sequence <= previous.sequence or current.observed_at <= previous.observed_at:
                raise ValueError("observation sequence and observedAt must increase strictly within each session")
            seconds = (current.observed_at - previous.observed_at).total_seconds()
            if seconds <= max_gap:
                if previous.score is None:
                    unknown += seconds
                else:
                    known += seconds
        duration = (session.end - session.start).total_seconds()
        total_known += known
        total_unknown += unknown
        coverage_sessions.append({
            "camera_id": session.camera_id,
            "source_epoch": session.source_epoch,
            "started_at": iso_time(session.start),
            "ended_at": iso_time(session.end),
            "annotated_seconds": duration,
            "known_seconds": known,
            "unknown_seconds": unknown,
            "unobserved_seconds": max(0.0, duration - known - unknown),
            "observation_count": len(rows),
        })

    matched_truths: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    ordered_starts = sorted(starts.values(), key=lambda item: (item[0].observed_at, item[0].event_id))
    for alarm, epoch, session_index in ordered_starts:
        candidates = [
            index for index, truth in enumerate(truths)
            if index not in matched_truths and truth_session[index] == session_index
            and (alarm.observed_at - truth.start).total_seconds() >= -tolerance
            and (alarm.observed_at - truth.end).total_seconds() <= tolerance
        ]
        identity = {
            "event_id": alarm.event_id, "camera_id": alarm.camera_id, "source_epoch": epoch,
            "alarm_observed_at": iso_time(alarm.observed_at),
        }
        if not candidates:
            false_positives.append(identity)
            continue
        # Earliest-ending eligible interval gives maximum one-to-one matching
        # for time-ordered points, including overlapping tolerance windows.
        selected = min(candidates, key=lambda index: (truths[index].end, truths[index].start))
        matched_truths.add(selected)
        truth = truths[selected]
        matches.append({
            **identity,
            "truth_event_id": truth.label,
            "truth_started_at": iso_time(truth.start),
            "truth_ended_at": iso_time(truth.end),
            "delay_seconds": (alarm.observed_at - truth.start).total_seconds(),
        })

    annotated_seconds = sum(item["annotated_seconds"] for item in coverage_sessions)
    normal_seconds = annotated_seconds - sum((truth.end - truth.start).total_seconds() for truth in truths)
    tp, fp, fn = len(matches), len(false_positives), len(truths) - len(matches)
    delays = [item["delay_seconds"] for item in matches]
    orphan_ids = sorted(other_ids - starts.keys())
    warnings = []
    if orphan_ids:
        warnings.append("UPDATE/END records without START cannot be scored as alarms; event input is incomplete")
    if total_known + total_unknown < annotated_seconds:
        warnings.append("observation coverage is incomplete; unobserved time is not known or normal inference")
    return {
        "schema_version": "1.0",
        "event_type": event_type,
        "matching": {
            "rule": "one-to-one START observedAt within truth interval +/- tolerance; same camera/epoch/session",
            "tolerance_seconds": tolerance,
            "duplicate_start_count": duplicate_starts,
            "ignored_other_type_record_count": ignored_other_type_records,
            "orphan_event_ids": orphan_ids,
        },
        "metrics": {
            "ground_truth_event_count": len(truths),
            "alarm_count": len(starts),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / len(truths) if truths else None,
            "miss_rate": fn / len(truths) if truths else None,
            "false_alarms_per_hour": fp / (annotated_seconds / 3600),
            "false_alarms_per_hour_denominator": "all explicitly annotated camera-session hours",
            "annotated_hours": annotated_seconds / 3600,
            "normal_hours": normal_seconds / 3600,
            "delay_seconds": {
                "count": len(delays), "mean": mean(delays) if delays else None,
                "median": median(delays) if delays else None,
                "min": min(delays) if delays else None, "max": max(delays) if delays else None,
                "definition": "START observedAt minus truth startedAt; negative allowed only with tolerance",
            },
        },
        "coverage": {
            "max_observation_gap_seconds": max_gap,
            "rule": "earlier observation state until next row when gap <= max; no leading/trailing extrapolation",
            "fraction_denominator": "all explicitly annotated camera-session seconds",
            "known_observation_count": known_count,
            "unknown_observation_count": unknown_count,
            "annotated_seconds": annotated_seconds,
            "known_seconds": total_known,
            "unknown_seconds": total_unknown,
            "unobserved_seconds": max(0.0, annotated_seconds - total_known - total_unknown),
            "known_fraction": total_known / annotated_seconds,
            "unknown_fraction": total_unknown / annotated_seconds,
            "unobserved_fraction": max(0.0, 1 - (total_known + total_unknown) / annotated_seconds),
            "complete": total_known + total_unknown == annotated_seconds,
            "sessions": coverage_sessions,
        },
        "matches": matches,
        "false_positive_alarms": false_positives,
        "missed_truth_event_ids": [truth.label for index, truth in enumerate(truths) if index not in matched_truths],
        "warnings": warnings,
    }
