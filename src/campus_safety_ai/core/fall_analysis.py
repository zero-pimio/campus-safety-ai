"""Conservative single-track fall event lifecycle.

A low score ends the observed falling *motion*. It does not establish that a
person has recovered or is safe. Unknown observations never count as low scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.contracts.models import iso_time, parse_time


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class FallObservation:
    camera_id: str
    source_epoch: int
    sequence: int
    observed_at: datetime
    window_started_at: datetime
    window_ended_at: datetime
    score: float | None
    model_version: str = "unknown"
    subject_track_keys: tuple[str, ...] = ()
    evidence_uris: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        for timestamp in (self.observed_at, self.window_started_at, self.window_ended_at):
            iso_time(timestamp)
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if not _nonnegative_integer(self.source_epoch) or not _nonnegative_integer(self.sequence):
            raise ValueError("source_epoch and sequence must be non-negative integers")
        if not self.window_started_at <= self.window_ended_at <= self.observed_at:
            raise ValueError("fall window must end after it starts and before observation")
        if self.score is not None and (not _finite_number(self.score) or not 0 <= self.score <= 1):
            raise ValueError("fall score must be a finite number between 0 and 1 or None")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")
        for values in (self.subject_track_keys, self.evidence_uris):
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError("track keys and evidence URIs must be tuples of non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameraId": self.camera_id,
            "sourceEpoch": self.source_epoch,
            "sequence": self.sequence,
            "observedAt": iso_time(self.observed_at),
            "windowStartedAt": iso_time(self.window_started_at),
            "windowEndedAt": iso_time(self.window_ended_at),
            "behavior": "falling",
            "score": self.score,
            "modelVersion": self.model_version,
            "subjectTrackKeys": list(self.subject_track_keys),
            "evidenceUris": list(self.evidence_uris),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FallObservation:
        for field in ("subjectTrackKeys", "evidenceUris"):
            if not isinstance(value.get(field, []), (list, tuple)):
                raise ValueError(f"{field} must be an array")
        return cls(
            camera_id=value["cameraId"],
            source_epoch=value["sourceEpoch"],
            sequence=value["sequence"],
            observed_at=parse_time(value["observedAt"]),
            window_started_at=parse_time(value["windowStartedAt"]),
            window_ended_at=parse_time(value["windowEndedAt"]),
            score=value.get("score"),
            model_version=value.get("modelVersion", "unknown"),
            subject_track_keys=tuple(value.get("subjectTrackKeys", [])),
            evidence_uris=tuple(value.get("evidenceUris", [])),
            reason=value.get("reason", ""),
        )


@dataclass(frozen=True)
class FallPolicy:
    edge_id: str
    start_score: float = 0.75
    end_score: float = 0.35
    confirm_seconds: float = 1.0
    clear_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    max_observation_gap_seconds: float = 1.0
    config_version: str = "fall-v1"
    event_type: str = "person_falling"
    max_cameras: int = 256
    max_evidence_uris: int = 8

    def __post_init__(self) -> None:
        for value in (self.edge_id, self.config_version, self.event_type):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("edge_id, config_version and event_type must be non-empty")
        if (not _finite_number(self.end_score) or not _finite_number(self.start_score)
                or not 0 <= self.end_score < self.start_score <= 1):
            raise ValueError("fall thresholds must satisfy 0 <= end_score < start_score <= 1")
        durations = (self.confirm_seconds, self.clear_seconds, self.cooldown_seconds)
        if any(not _finite_number(value) or value < 0 for value in durations):
            raise ValueError("fall durations must be finite and non-negative")
        if not _finite_number(self.max_observation_gap_seconds) or self.max_observation_gap_seconds <= 0:
            raise ValueError("max_observation_gap_seconds must be finite and positive")
        if any(not _nonnegative_integer(value) or value < 1 for value in (self.max_cameras, self.max_evidence_uris)):
            raise ValueError("state limits must be positive integers")


@dataclass
class _FallState:
    source_epoch: int
    last_sequence: int = -1
    last_observed_at: datetime | None = None
    finalized: bool = False
    track_keys: tuple[str, ...] = ()
    positive_since: datetime | None = None
    negative_since: datetime | None = None
    cooldown_until: datetime | None = None
    event_id: str | None = None
    started_at: datetime | None = None
    revision: int = 0
    evidence_uris: tuple[str, ...] = ()
    last_score: float = 0.0
    model_version: str = "unknown"


class FallEventAnalysis:
    """Use score cadence, hysteresis and source boundaries to emit event records.

    One state per camera and capped evidence lists keep memory bounded. Epochs
    must increase on reconnect. Finalization leaves a small epoch watermark so
    duplicate input cannot reopen a completed stream. ``release_camera`` drops
    that watermark after finalization when a camera is permanently removed.

    ``last_record_reasons`` contains only the latest call's record reasons for
    inclusion in an outer batch, without extending the EventRecord contract.
    """

    def __init__(self, policy: FallPolicy) -> None:
        self.policy = policy
        self._states: dict[str, _FallState] = {}
        self.last_record_reasons: dict[str, str] = {}
        self.last_reason: str | None = None

    @property
    def state_count(self) -> int:
        return len(self._states)

    def advance(self, observation: FallObservation) -> list[EventRecord]:
        self._reset_reasons()
        state = self._states.get(observation.camera_id)
        records: list[EventRecord] = []
        if state is not None:
            if observation.source_epoch < state.source_epoch:
                return []
            if observation.source_epoch == state.source_epoch:
                if state.finalized or observation.sequence <= state.last_sequence:
                    return []
                if state.last_observed_at is not None and observation.observed_at <= state.last_observed_at:
                    return []
            else:
                if state.last_observed_at is not None and observation.observed_at < state.last_observed_at:
                    raise ValueError("new source epoch cannot precede the previous observation")
                records.extend(self._close(
                    observation.camera_id, state, observation.observed_at,
                    "source_epoch_changed", ended_at=state.last_observed_at,
                ))
                state = None
        if state is None:
            if observation.camera_id not in self._states and len(self._states) >= self.policy.max_cameras:
                raise ValueError("camera limit reached; finalize and release an unused camera")
            state = _FallState(source_epoch=observation.source_epoch)
            self._states[observation.camera_id] = state

        if state.last_observed_at is not None:
            gap = (observation.observed_at - state.last_observed_at).total_seconds()
            if gap > self.policy.max_observation_gap_seconds:
                records.extend(self._close(
                    observation.camera_id, state, observation.observed_at,
                    "observation_gap", ended_at=state.last_observed_at,
                ))
                self._reset_runs(state)

        # Missing/ambiguous subjects are unknown, not evidence of a new identity.
        # A newly known track (even during warm-up) is an explicit boundary.
        single_track = len(observation.subject_track_keys) == 1
        if single_track and state.track_keys and state.track_keys != observation.subject_track_keys:
            records.extend(self._close(
                observation.camera_id, state, observation.observed_at, "track_changed",
            ))
            self._reset_runs(state)
            state.evidence_uris = ()
        if single_track:
            state.track_keys = observation.subject_track_keys

        state.last_sequence = observation.sequence
        state.last_observed_at = observation.observed_at
        if observation.score is None or not single_track:
            self._reset_runs(state)
            return records

        state.last_score = observation.score
        state.model_version = observation.model_version
        if state.event_id is not None:
            state.evidence_uris = self._evidence(state.evidence_uris, observation.evidence_uris)

        if observation.score >= self.policy.start_score:
            state.negative_since = None
            if state.event_id is not None:
                return records
            if state.cooldown_until is not None and observation.observed_at < state.cooldown_until:
                state.positive_since = None
                return records
            state.positive_since = state.positive_since or observation.observed_at
            state.evidence_uris = self._evidence(state.evidence_uris, observation.evidence_uris)
            if (observation.observed_at - state.positive_since).total_seconds() >= self.policy.confirm_seconds:
                state.started_at = state.positive_since
                identity = (
                    f"{observation.camera_id}:{observation.source_epoch}:"
                    f"{self.policy.event_type}:{state.track_keys}:{iso_time(state.started_at)}"
                )
                state.event_id = str(uuid5(NAMESPACE_URL, identity))
                state.revision = 1
                record = self._record(observation.camera_id, state, observation.observed_at, "START")
                self.last_record_reasons[record.idempotency_key] = "confirmed"
                records.append(record)
            return records

        state.positive_since = None
        if observation.score <= self.policy.end_score and state.event_id is not None:
            state.negative_since = state.negative_since or observation.observed_at
            if (observation.observed_at - state.negative_since).total_seconds() >= self.policy.clear_seconds:
                records.extend(self._close(
                    observation.camera_id, state, observation.observed_at, "motion_cleared",
                ))
        else:
            state.negative_since = None
            if state.event_id is None:
                state.evidence_uris = ()
        return records

    def finalize(
        self, camera_id: str, source_epoch: int, ended_at: datetime, reason: str = "source_end",
    ) -> list[EventRecord]:
        """Close the current epoch; an END here does not establish recovery."""
        self._reset_reasons()
        iso_time(ended_at)
        if not isinstance(camera_id, str) or not camera_id.strip() or not _nonnegative_integer(source_epoch):
            raise ValueError("finalize requires a camera_id and a non-negative integer source_epoch")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("finalization reason must be non-empty")
        state = self._states.get(camera_id)
        if state is None or state.source_epoch != source_epoch or state.finalized:
            return []
        if state.last_observed_at is not None and ended_at < state.last_observed_at:
            raise ValueError("finalization cannot precede the last observation")
        records = self._close(camera_id, state, ended_at, reason)
        self._states[camera_id] = _FallState(
            source_epoch=source_epoch, last_sequence=state.last_sequence,
            last_observed_at=ended_at, finalized=True,
        )
        return records

    def release_camera(self, camera_id: str) -> None:
        """Forget a finalized camera, including its duplicate-input watermark."""
        state = self._states.get(camera_id)
        if state is not None and not state.finalized:
            raise ValueError("finalize the camera before releasing its state")
        self._states.pop(camera_id, None)

    def _reset_reasons(self) -> None:
        self.last_record_reasons = {}
        self.last_reason = None

    @staticmethod
    def _reset_runs(state: _FallState) -> None:
        state.positive_since = None
        state.negative_since = None
        if state.event_id is None:
            state.evidence_uris = ()

    def _evidence(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*left, *right)))[-self.policy.max_evidence_uris:]

    def _close(
        self, camera_id: str, state: _FallState, observed_at: datetime, reason: str,
        *, ended_at: datetime | None = None,
    ) -> list[EventRecord]:
        if state.event_id is None:
            return []
        state.revision += 1
        record = self._record(camera_id, state, observed_at, "END", ended_at=ended_at)
        self.last_record_reasons[record.idempotency_key] = reason
        self.last_reason = reason
        state.event_id = None
        state.started_at = None
        state.evidence_uris = ()
        self._reset_runs(state)
        state.cooldown_until = observed_at + timedelta(seconds=self.policy.cooldown_seconds)
        return [record]

    def _record(
        self, camera_id: str, state: _FallState, observed_at: datetime,
        phase: Literal["START", "END"], *, ended_at: datetime | None = None,
    ) -> EventRecord:
        assert state.event_id is not None and state.started_at is not None
        return EventRecord(
            schema_version="1.0", event_id=state.event_id, revision=state.revision,
            phase=phase, event_type=self.policy.event_type, severity="critical",
            edge_id=self.policy.edge_id, camera_id=camera_id,
            started_at=state.started_at, observed_at=observed_at,
            ended_at=(ended_at or observed_at) if phase == "END" else None,
            subject_track_keys=state.track_keys, confidence=state.last_score,
            model_version=state.model_version, config_version=self.policy.config_version,
            idempotency_key=f"{state.event_id}:{state.revision}",
            status="CLOSED" if phase == "END" else "OPEN", evidence_uris=state.evidence_uris,
        )
