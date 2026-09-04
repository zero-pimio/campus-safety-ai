from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from campus_safety_ai.contracts import BBox, Detection, Detections, EventRecord, Track


def _iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _inside_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
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

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError("intrusion polygon needs at least three points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError("intrusion polygon points must be normalized to [0, 1]")


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
        if self._epoch != epoch:
            self._objects.clear()
            self._next_id = 1
            self._epoch = epoch

        expired = tuple(
            self._track_key(batch.camera_id, batch.source_epoch, key)
            for key, value in self._objects.items()
            if (batch.captured_at - value.last_seen).total_seconds() > self.max_gap_seconds
        )
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
            if score < self.iou_threshold:
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
        return TrackingResult(tuple(result), expired)


@dataclass
class _IntrusionState:
    entered_at: datetime | None = None
    outside_since: datetime | None = None
    cooldown_until: datetime | None = None
    event_id: str | None = None
    started_at: datetime | None = None
    revision: int = 0


class EventAnalysis:
    def __init__(self, policy: IntrusionPolicy, tracker: Tracker | None = None) -> None:
        self.policy = policy
        self.tracker = tracker or SimpleIoUTracker()
        self._states: dict[str, _IntrusionState] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}

    def advance(self, batch: Detections) -> list[EventRecord]:
        stream = (batch.camera_id, batch.source_epoch)
        previous = self._last_sequence.get(stream)
        if previous is not None and batch.sequence <= previous:
            return []
        self._last_sequence[stream] = batch.sequence

        records: list[EventRecord] = []
        tracking = self.tracker.update(batch)
        for track_key in tracking.expired_track_keys:
            state = self._states.pop(track_key, None)
            if state is None or state.event_id is None:
                continue
            state.revision += 1
            records.append(
                self._ending_record(batch.camera_id, batch.captured_at, track_key, 0.0, state)
            )
        for track in tracking.tracks:
            if track.label != "person" or track.confidence < self.policy.minimum_confidence:
                continue
            state = self._states.setdefault(track.track_key, _IntrusionState())
            point = track.bbox.bottom_center_normalized(batch.width, batch.height)
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
        prefix = f"{camera_id}:{source_epoch}:"
        records: list[EventRecord] = []
        for key in sorted(self._states):
            if not key.startswith(prefix):
                continue
            state = self._states[key]
            if state.event_id is None:
                continue
            state.revision += 1
            records.append(
                self._ending_record(camera_id, ended_at, key, 0.0, state)
            )
            state.cooldown_until = ended_at + timedelta(seconds=self.policy.cooldown_seconds)
            state.event_id = None
            state.started_at = None
            state.entered_at = None
            state.outside_since = None
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
            event_type="intrusion",
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
            event_type="intrusion",
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
