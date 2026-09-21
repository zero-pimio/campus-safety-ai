from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from campus_safety_ai.contracts import BBox, Detections, EventRecord, Track


def _finite_nonnegative(name: str, value: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _aware_time(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")


def _validate_batch(batch: Detections) -> None:
    _aware_time(batch.captured_at)
    for name in ("source_epoch", "sequence", "width", "height"):
        value = getattr(batch, name)
        minimum = 1 if name in ("width", "height") else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not batch.camera_id:
        raise ValueError("camera_id must not be empty")
    for detection in batch.detections:
        if not all(math.isfinite(value) for value in detection.bbox.to_list()):
            raise ValueError("detection bbox coordinates must be finite")


def _iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _inside_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Ray casting with polygon edges and vertices included (normalized epsilon 1e-12)."""
    x, y = point
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-12
            and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
        ):
            return True
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class IntrusionPolicy:
    edge_id: str
    zone_id: str
    polygon: tuple[tuple[float, float], ...]
    enter_seconds: float = 2.0
    exit_seconds: float = 1.0
    cooldown_seconds: float = 10.0
    minimum_confidence: float = 0.4
    config_version: str = "intrusion-v1"
    event_type: str = "intrusion"
    target_label: str = "person"
    max_observation_gap_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("edge_id", "zone_id", "config_version", "event_type", "target_label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("enter_seconds", "exit_seconds", "cooldown_seconds", "max_observation_gap_seconds"):
            _finite_nonnegative(name, getattr(self, name))
        _finite_nonnegative("minimum_confidence", self.minimum_confidence, 1.0)
        if len(self.polygon) < 3:
            raise ValueError("intrusion polygon needs at least three points")
        for x, y in self.polygon:
            _finite_nonnegative("polygon x", x, 1.0)
            _finite_nonnegative("polygon y", y, 1.0)
        area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(self.polygon, (*self.polygon[1:], self.polygon[0]), strict=True)
        )
        if abs(area) <= 1e-12:
            raise ValueError("intrusion polygon must enclose a non-zero area")


@dataclass
class _TrackedObject:
    local_id: int
    bbox: BBox
    label: str
    confidence: float
    last_seen: datetime


@dataclass(frozen=True)
class TrackingResult:
    tracks: tuple[Track, ...]
    expired_track_keys: tuple[str, ...]


class Tracker(Protocol):
    def update(self, batch: Detections) -> TrackingResult: ...


class SimpleIoUTracker:
    """Deterministic G1 tracker; replace internally with ByteTrack for G3."""

    def __init__(self, iou_threshold: float = 0.25, max_gap_seconds: float = 2.0) -> None:
        _finite_nonnegative("iou_threshold", iou_threshold, 1.0)
        _finite_nonnegative("max_gap_seconds", max_gap_seconds)
        self.iou_threshold = iou_threshold
        self.max_gap_seconds = max_gap_seconds
        self._next_id = 1
        self._objects: dict[int, _TrackedObject] = {}
        self._epoch: tuple[str, int] | None = None

    @staticmethod
    def _track_key(camera_id: str, source_epoch: int, local_id: int) -> str:
        return f"{camera_id}:{source_epoch}:1:{local_id}"

    def update(self, batch: Detections) -> TrackingResult:
        epoch = (batch.camera_id, batch.source_epoch)
        expired: list[str] = []
        if self._epoch != epoch:
            if self._epoch is not None:
                # Tracks cannot cross a source epoch (CONTEXT.md); report them so
                # open events held by vanished tracks get closed instead of leaked.
                old_camera, old_epoch = self._epoch
                expired.extend(
                    self._track_key(old_camera, old_epoch, key) for key in sorted(self._objects)
                )
            self._objects.clear()
            self._next_id = 1
            self._epoch = epoch
        for key, value in self._objects.items():
            if (batch.captured_at - value.last_seen).total_seconds() > self.max_gap_seconds:
                expired.append(self._track_key(batch.camera_id, batch.source_epoch, key))
        self._objects = {
            key: value
            for key, value in self._objects.items()
            if (batch.captured_at - value.last_seen).total_seconds() <= self.max_gap_seconds
        }
        unmatched = set(self._objects)
        result: list[Track] = []
        for detection in sorted(batch.detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                (track_id, _iou(detection.bbox, tracked.bbox))
                for track_id, tracked in self._objects.items()
                if track_id in unmatched and tracked.label == detection.label
            ]
            track_id, score = max(candidates, key=lambda item: item[1], default=(0, 0.0))
            if track_id == 0 or score < self.iou_threshold:
                track_id = self._next_id
                self._next_id += 1
            else:
                unmatched.remove(track_id)
            self._objects[track_id] = _TrackedObject(
                track_id, detection.bbox, detection.label, detection.confidence, batch.captured_at
            )
            result.append(
                Track(
                    track_key=self._track_key(batch.camera_id, batch.source_epoch, track_id),
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    observed_at=batch.captured_at,
                )
            )
        return TrackingResult(tuple(result), tuple(expired))

    def finalize(self, camera_id: str, source_epoch: int) -> None:
        if self._epoch == (camera_id, source_epoch):
            self._objects.clear()
            self._epoch = None
            self._next_id = 1


@dataclass
class _IntrusionState:
    camera_id: str = ""
    source_epoch: int = 0
    last_valid_at: datetime | None = None
    entered_at: datetime | None = None
    outside_since: datetime | None = None
    cooldown_until: datetime | None = None
    event_id: str | None = None
    started_at: datetime | None = None
    revision: int = 0


class EventAnalysis:
    """One active source per tracker; use a separate instance for each camera."""

    def __init__(self, policy: IntrusionPolicy, tracker: Tracker | None = None) -> None:
        self.policy = policy
        self.tracker = tracker or SimpleIoUTracker(max_gap_seconds=policy.max_observation_gap_seconds)
        self._states: dict[str, _IntrusionState] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._last_captured_at: dict[tuple[str, int], datetime] = {}
        # One watermark per camera, not one entry per historical epoch.
        self._source_epochs: dict[str, tuple[int, bool]] = {}
        self._active_stream: tuple[str, int] | None = None

    def advance(self, batch: Detections) -> list[EventRecord]:
        _validate_batch(batch)
        stream = (batch.camera_id, batch.source_epoch)
        watermark = self._source_epochs.get(batch.camera_id)
        if watermark is not None and (
            batch.source_epoch < watermark[0] or (batch.source_epoch == watermark[0] and watermark[1])
        ):
            return []
        previous = self._last_sequence.get(stream)
        if previous is not None and batch.sequence <= previous:
            return []
        previous_time = self._last_captured_at.get(stream)
        if previous_time is not None and batch.captured_at < previous_time:
            raise ValueError("captured_at must not go backwards within a source epoch")

        records: list[EventRecord] = []
        if self._active_stream is not None and self._active_stream != stream:
            old_camera, old_epoch = self._active_stream
            old_time = self._last_captured_at[self._active_stream]
            records.extend(self.finalize(old_camera, old_epoch, max(batch.captured_at, old_time)))
        # ByteTrack can associate across classes; gate semantic targets before
        # tracking, while retaining low-confidence target boxes for association.
        target_batch = replace(batch, detections=tuple(
            detection for detection in batch.detections if detection.label == self.policy.target_label
        ))
        tracking = self.tracker.update(target_batch)
        self._last_sequence[stream] = batch.sequence
        self._last_captured_at[stream] = batch.captured_at
        self._source_epochs[batch.camera_id] = (batch.source_epoch, False)
        self._active_stream = stream
        for track_key in tracking.expired_track_keys:
            state = self._states.pop(track_key, None)
            if state is None or state.event_id is None:
                continue
            state.revision += 1
            records.append(
                self._ending_record(state.camera_id, batch.captured_at, track_key, 0.0, state)
            )
        valid_tracks = {
            track.track_key: track for track in tracking.tracks
            if track.label == self.policy.target_label
            and math.isfinite(track.confidence) and track.confidence >= self.policy.minimum_confidence
            and track.observed_at == batch.captured_at
        }
        for track_key, state in list(self._states.items()):
            gap = ((batch.captured_at - state.last_valid_at).total_seconds()
                   if state.last_valid_at is not None else math.inf)
            if gap > self.policy.max_observation_gap_seconds:
                if state.event_id is not None:
                    state.revision += 1
                    records.append(self._ending_record(state.camera_id, batch.captured_at, track_key, 0.0, state))
                    state.cooldown_until = batch.captured_at + timedelta(seconds=self.policy.cooldown_seconds)
                    state.event_id = None
                    state.started_at = None
                state.entered_at = None
                state.outside_since = None
                if state.cooldown_until is None or batch.captured_at >= state.cooldown_until:
                    del self._states[track_key]
            elif track_key not in valid_tracks:
                # Unknown is neither continued presence nor a verified departure.
                if state.event_id is None:
                    state.entered_at = None
                state.outside_since = None

        for track in valid_tracks.values():
            point = track.bbox.bottom_center_normalized(batch.width, batch.height)
            if not all(math.isfinite(value) for value in point):
                continue
            state = self._states.setdefault(
                track.track_key, _IntrusionState(camera_id=batch.camera_id, source_epoch=batch.source_epoch)
            )
            state.last_valid_at = batch.captured_at
            inside = _inside_polygon(point, self.policy.polygon)
            if inside:
                state.outside_since = None
                if state.entered_at is None:
                    state.entered_at = batch.captured_at
                ready = (batch.captured_at - state.entered_at).total_seconds() >= self.policy.enter_seconds
                cooldown_over = state.cooldown_until is None or batch.captured_at >= state.cooldown_until
                if ready and cooldown_over and state.event_id is None:
                    key = f"{batch.camera_id}:intrusion:{self.policy.zone_id}:{track.track_key}:{state.entered_at.isoformat()}"
                    state.event_id = str(uuid5(NAMESPACE_URL, key))
                    state.started_at = state.entered_at
                    state.revision = 1
                    records.append(self._record(batch, track, state, "START"))
            else:
                state.entered_at = None if state.event_id is None else state.entered_at
                if state.event_id is not None:
                    state.outside_since = state.outside_since or batch.captured_at
                    if (batch.captured_at - state.outside_since).total_seconds() >= self.policy.exit_seconds:
                        records.append(self._close(batch, track, state))
        return records

    def finalize(
        self, camera_id: str, source_epoch: int, ended_at: datetime
    ) -> list[EventRecord]:
        """Close every open event for one source epoch when its stream ends.

        Tracks cannot survive a new source epoch (see CONTEXT.md), so a stream
        restart or replay end must never leave an event stuck in the OPEN state.
        """
        _aware_time(ended_at)
        stream = (camera_id, source_epoch)
        previous_time = self._last_captured_at.get(stream)
        if previous_time is not None and ended_at < previous_time:
            raise ValueError("ended_at must not precede the latest captured_at")
        records: list[EventRecord] = []
        for key in sorted(self._states):
            if (self._states[key].camera_id, self._states[key].source_epoch) != stream:
                continue
            # The epoch is over: closed states can never produce events again,
            # so dropping them keeps long-running services from growing forever.
            state = self._states.pop(key)
            if state.event_id is None:
                continue
            state.revision += 1
            records.append(
                self._ending_record(camera_id, ended_at, key, 0.0, state)
            )
        self._last_sequence.pop(stream, None)
        self._last_captured_at.pop(stream, None)
        watermark = self._source_epochs.get(camera_id)
        if watermark is not None and watermark[0] == source_epoch:
            self._source_epochs[camera_id] = (source_epoch, True)
        if self._active_stream == stream:
            self._active_stream = None
        tracker_finalize = getattr(self.tracker, "finalize", None)
        if callable(tracker_finalize):
            tracker_finalize(camera_id, source_epoch)
        return records

    def _ending_record(
        self,
        camera_id: str,
        ended_at: datetime,
        track_key: str,
        confidence: float,
        state: _IntrusionState,
    ) -> EventRecord:
        assert state.event_id is not None and state.started_at is not None
        return EventRecord(
            schema_version="1.0",
            event_id=state.event_id,
            revision=state.revision,
            phase="END",
            event_type=self.policy.event_type,
            severity="warning",
            edge_id=self.policy.edge_id,
            camera_id=camera_id,
            started_at=state.started_at,
            observed_at=ended_at,
            ended_at=ended_at,
            subject_track_keys=(track_key,),
            confidence=confidence,
            model_version="stream-finalize",
            config_version=self.policy.config_version,
            idempotency_key=f"{state.event_id}:{state.revision}",
            status="CLOSED",
        )

    def _close(self, batch: Detections, track: Track, state: _IntrusionState) -> EventRecord:
        state.revision += 1
        record = self._record(batch, track, state, "END")
        state.cooldown_until = batch.captured_at + timedelta(seconds=self.policy.cooldown_seconds)
        state.event_id = None
        state.started_at = None
        state.entered_at = None
        state.outside_since = None
        return record

    def _record(
        self, batch: Detections, track: Track, state: _IntrusionState, phase: str
    ) -> EventRecord:
        assert state.event_id is not None and state.started_at is not None
        return EventRecord(
            schema_version="1.0",
            event_id=state.event_id,
            revision=state.revision,
            phase=phase,  # type: ignore[arg-type]
            event_type=self.policy.event_type,
            severity="warning",
            edge_id=self.policy.edge_id,
            camera_id=batch.camera_id,
            started_at=state.started_at,
            observed_at=batch.captured_at,
            ended_at=batch.captured_at if phase == "END" else None,
            subject_track_keys=(track.track_key,),
            confidence=track.confidence,
            model_version=batch.model_version,
            config_version=self.policy.config_version,
            idempotency_key=f"{state.event_id}:{state.revision}",
            status="CLOSED" if phase == "END" else "OPEN",
        )
