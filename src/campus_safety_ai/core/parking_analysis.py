"""Image-space stationary vehicle events inside a configured no-parking zone."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from campus_safety_ai.contracts import Detections, EventRecord, Track, iso_time
from campus_safety_ai.core.event_analysis import SimpleIoUTracker, Tracker, _inside_polygon


@dataclass(frozen=True)
class ParkingPolicy:
    edge_id: str
    zone_id: str
    polygon: tuple[tuple[float, float], ...]
    target_labels: tuple[str, ...] = ("car", "truck", "bus", "motorcycle")
    minimum_confidence: float = 0.4
    stationary_seconds: float = 30.0
    movement_threshold: float = 0.02
    exit_seconds: float = 2.0
    cooldown_seconds: float = 30.0
    max_observation_gap_seconds: float = 2.0
    config_version: str = "parking-v1"

    def __post_init__(self) -> None:
        for name in ("edge_id", "zone_id", "config_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.target_labels, str) or not self.target_labels or any(not isinstance(label, str) or not label.strip()
                                         for label in self.target_labels):
            raise ValueError("parking target_labels must contain non-empty labels")
        if len(set(self.target_labels)) != len(self.target_labels):
            raise ValueError("parking target_labels must be unique")
        for name in ("minimum_confidence", "stationary_seconds", "movement_threshold", "exit_seconds",
                     "cooldown_seconds", "max_observation_gap_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_confidence > 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.stationary_seconds == 0 or self.max_observation_gap_seconds == 0:
            raise ValueError("stationary_seconds and max_observation_gap_seconds must be positive")
        if self.movement_threshold > math.sqrt(2):
            raise ValueError("movement_threshold must not exceed the normalized image diagonal")
        if len(self.polygon) < 3:
            raise ValueError("parking polygon needs at least three points")
        for point in self.polygon:
            if len(point) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                      or not 0 <= value <= 1 for value in point):
                raise ValueError("parking polygon points must be finite and normalized to [0, 1]")
        area = sum(left[0] * right[1] - right[0] * left[1]
                   for left, right in zip(self.polygon, self.polygon[1:] + self.polygon[:1], strict=True))
        if abs(area) <= 1e-12:
            raise ValueError("parking polygon must have non-zero area")


@dataclass
class _Stream:
    source_epoch: int
    sequence: int
    captured_at: datetime
    finalized: bool = False


@dataclass
class _ParkingState:
    camera_id: str
    source_epoch: int
    track_key: str
    last_seen: datetime
    model_version: str
    confidence: float
    anchor: tuple[float, float] | None = None
    stationary_since: datetime | None = None
    departure_since: datetime | None = None
    cooldown_until: datetime | None = None
    event_id: str | None = None
    started_at: datetime | None = None
    revision: int = 0


class ParkingAnalysis:
    """Detect sustained image-space stationarity; no world-speed inference."""

    def __init__(self, policy: ParkingPolicy, tracker: Tracker | None = None) -> None:
        self.policy = policy
        self.tracker = tracker
        self._trackers: dict[str, SimpleIoUTracker] = {}
        self._states: dict[str, _ParkingState] = {}
        # One watermark per camera, rather than one entry per retired source epoch.
        self._streams: dict[str, _Stream] = {}

    def advance(self, batch: Detections) -> list[EventRecord]:
        self._validate_batch(batch)
        stream = self._streams.get(batch.camera_id)
        if stream is not None:
            if batch.source_epoch < stream.source_epoch:
                return []
            if batch.source_epoch == stream.source_epoch and (
                stream.finalized or batch.sequence <= stream.sequence or batch.captured_at <= stream.captured_at
            ):
                return []
        records: list[EventRecord] = []
        if stream is not None and batch.source_epoch > stream.source_epoch:
            records.extend(self.finalize(batch.camera_id, stream.source_epoch, batch.captured_at))
        self._streams[batch.camera_id] = _Stream(batch.source_epoch, batch.sequence, batch.captured_at)
        tracker = self.tracker
        if tracker is None:
            tracker = self._trackers.setdefault(batch.camera_id, SimpleIoUTracker(
                max_gap_seconds=self.policy.max_observation_gap_seconds,
            ))
        # Trackers such as ByteTrack may associate boxes without class gating.
        # Keep non-vehicles out, but retain low-confidence vehicles for tracking.
        tracking = tracker.update(replace(
            batch, detections=tuple(item for item in batch.detections if item.label in self.policy.target_labels),
        ))
        for key in tracking.expired_track_keys:
            state = self._states.pop(key, None)
            if state is not None and state.event_id is not None:
                records.append(self._end(state, batch.captured_at, 0.0, "track-expired"))
        for key, state in list(self._states.items()):
            if self._same_stream(state, batch) and (
                batch.captured_at - state.last_seen
            ).total_seconds() > self.policy.max_observation_gap_seconds:
                if (state.event_id is None and state.cooldown_until is not None
                        and batch.captured_at < state.cooldown_until):
                    # A weak detection can keep the tracker identity alive without
                    # supplying valid dwell evidence. Its existing cooldown still
                    # applies, and expires normally even if no strong track returns.
                    self._reset_candidate(state)
                    continue
                self._states.pop(key)
                if state.event_id is not None:
                    records.append(self._end(state, batch.captured_at, 0.0, "observation-gap"))

        observed: set[str] = set()
        for track in tracking.tracks:
            if track.track_key in observed or not self._eligible(track, batch):
                continue
            observed.add(track.track_key)
            point = track.bbox.bottom_center_normalized(batch.width, batch.height)
            inside = _inside_polygon(point, self.policy.polygon)
            state = self._states.get(track.track_key)
            if state is None:
                if not inside:
                    continue
                state = _ParkingState(batch.camera_id, batch.source_epoch, track.track_key,
                                      batch.captured_at, batch.model_version, track.confidence)
                self._states[track.track_key] = state
            state.last_seen = batch.captured_at
            state.confidence = track.confidence
            state.model_version = batch.model_version
            if state.event_id is not None:
                moved = state.anchor is None or math.dist(point, state.anchor) > self.policy.movement_threshold + 1e-12
                if inside and not moved:
                    state.departure_since = None
                else:
                    state.departure_since = state.departure_since or batch.captured_at
                    if (batch.captured_at - state.departure_since).total_seconds() >= self.policy.exit_seconds:
                        records.append(self._end(state, batch.captured_at, track.confidence, batch.model_version))
                continue
            if not inside or (state.cooldown_until is not None and batch.captured_at < state.cooldown_until):
                self._reset_candidate(state)
                if state.cooldown_until is None or batch.captured_at >= state.cooldown_until:
                    self._states.pop(track.track_key)
                continue
            if state.anchor is None or math.dist(point, state.anchor) > self.policy.movement_threshold + 1e-12:
                state.anchor = point
                state.stationary_since = batch.captured_at
            assert state.stationary_since is not None
            if (batch.captured_at - state.stationary_since).total_seconds() >= self.policy.stationary_seconds:
                identity = (f"{self.policy.edge_id}:{state.camera_id}:illegal_parking:{self.policy.zone_id}:"
                            f"{state.track_key}:{state.stationary_since.isoformat()}")
                state.event_id = str(uuid5(NAMESPACE_URL, identity))
                state.started_at = state.stationary_since
                state.revision = 1
                records.append(self._record(state, "START", batch.captured_at, track.confidence, batch.model_version))

        for key, state in list(self._states.items()):
            if self._same_stream(state, batch) and key not in observed:
                # An empty/low-confidence observation breaks continuous dwell and
                # departure evidence, even when the tracker retains the identity.
                state.departure_since = None
                if state.event_id is None:
                    self._reset_candidate(state)
                    if state.cooldown_until is None or batch.captured_at >= state.cooldown_until:
                        self._states.pop(key)
        return records

    def finalize(self, camera_id: str, source_epoch: int, ended_at: datetime) -> list[EventRecord]:
        iso_time(ended_at)
        records = []
        for key, state in list(self._states.items()):
            if state.camera_id == camera_id and state.source_epoch == source_epoch:
                self._states.pop(key)
                if state.event_id is not None:
                    records.append(self._end(state, ended_at, 0.0, "stream-finalize"))
        stream = self._streams.get(camera_id)
        if stream is not None and stream.source_epoch == source_epoch:
            stream.finalized = True
            tracker = self.tracker if self.tracker is not None else self._trackers.get(camera_id)
            finalize_tracker = getattr(tracker, "finalize", None)
            if callable(finalize_tracker):
                finalize_tracker(camera_id, source_epoch)
            self._trackers.pop(camera_id, None)
        return records

    @staticmethod
    def _same_stream(state: _ParkingState, batch: Detections) -> bool:
        return state.camera_id == batch.camera_id and state.source_epoch == batch.source_epoch

    @staticmethod
    def _reset_candidate(state: _ParkingState) -> None:
        state.anchor = None
        state.stationary_since = None
        state.departure_since = None

    def _eligible(self, track: Track, batch: Detections) -> bool:
        return (track.track_key.startswith(f"{batch.camera_id}:{batch.source_epoch}:")
                and track.observed_at == batch.captured_at
                and track.label in self.policy.target_labels
                and math.isfinite(track.confidence)
                and self.policy.minimum_confidence <= track.confidence <= 1
                and all(math.isfinite(value) for value in track.bbox.to_list()))

    @staticmethod
    def _validate_batch(batch: Detections) -> None:
        iso_time(batch.captured_at)
        if not batch.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        for name in ("source_epoch", "sequence"):
            value = getattr(batch, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if batch.width <= 0 or batch.height <= 0 or not math.isfinite(batch.width) or not math.isfinite(batch.height):
            raise ValueError("frame dimensions must be finite and positive")
        if any(not math.isfinite(value) for item in batch.detections for value in item.bbox.to_list()):
            raise ValueError("detection coordinates must be finite")

    def _end(self, state: _ParkingState, ended_at: datetime, confidence: float, model_version: str) -> EventRecord:
        assert state.started_at is not None
        ended_at = max(ended_at, state.last_seen, state.started_at)
        state.revision += 1
        record = self._record(state, "END", ended_at, confidence, model_version)
        state.event_id = None
        state.started_at = None
        state.cooldown_until = ended_at + timedelta(seconds=self.policy.cooldown_seconds)
        self._reset_candidate(state)
        return record

    def _record(self, state: _ParkingState, phase: str, observed_at: datetime,
                confidence: float, model_version: str) -> EventRecord:
        assert state.event_id is not None and state.started_at is not None
        return EventRecord(
            schema_version="1.0", event_id=state.event_id, revision=state.revision,
            phase=phase,  # type: ignore[arg-type]
            event_type="illegal_parking", severity="warning", edge_id=self.policy.edge_id,
            camera_id=state.camera_id, started_at=state.started_at, observed_at=observed_at,
            ended_at=observed_at if phase == "END" else None, subject_track_keys=(state.track_key,),
            confidence=confidence, model_version=model_version, config_version=self.policy.config_version,
            idempotency_key=f"{state.event_id}:{state.revision}", status="CLOSED" if phase == "END" else "OPEN",
        )
