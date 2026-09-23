"""Bounded, causal fixed-duration pose windows for one conservatively tracked person."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from campus_safety_ai.contracts import EventRecord, iso_time
from campus_safety_ai.core.fall_analysis import FallEventAnalysis, FallObservation


@dataclass(frozen=True)
class PoseFrame:
    camera_id: str
    source_epoch: int
    sequence: int
    observed_at: datetime
    keypoints: Any
    box: Any
    track_key: str | None
    valid: bool
    reason: str = ""

    def __post_init__(self):
        iso_time(self.observed_at)
        if not self.camera_id.strip() or type(self.source_epoch) is not int or self.source_epoch < 1:
            raise ValueError("camera and positive source_epoch are required")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.valid and not self.track_key:
            raise ValueError("valid pose frames require a track key")


@dataclass(frozen=True)
class WindowPolicy:
    window_seconds: float = 1.0
    stride_seconds: float = 0.1
    sample_count: int = 16
    max_frame_gap_seconds: float = 0.3
    max_buffer_frames: int = 256

    def __post_init__(self):
        for value in (self.window_seconds, self.stride_seconds, self.max_frame_gap_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("window, stride and maximum gap must be finite and positive")
        if type(self.sample_count) is not int or self.sample_count < 4:
            raise ValueError("sample_count must be an integer >= 4")
        if type(self.max_buffer_frames) is not int or self.max_buffer_frames < self.sample_count + 1:
            raise ValueError("buffer must hold at least sample_count + 1 frames")


class PoseHead(Protocol):
    model_version: str

    def predict(self, frames: tuple[PoseFrame, ...]) -> tuple[float | None, dict[str, Any]]: ...


@dataclass(frozen=True)
class FallBatch:
    observation: FallObservation | None = None
    events: tuple[EventRecord, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)
    event_reasons: dict[str, str] = field(default_factory=dict)
    source_epoch: int = 1


class FallPipeline:
    """One source per instance. Never stores a full stream or future frames.

    All raw frames must enter advance, including invalid frames between inference
    ticks. A loss or identity break empties history; no descriptor bridges it.
    """

    def __init__(self, head: PoseHead, analysis: FallEventAnalysis, policy: WindowPolicy):
        if policy.stride_seconds > analysis.policy.max_observation_gap_seconds:
            raise ValueError("inference stride exceeds event observation gap")
        trained_window = getattr(head, "extraction_identity", {}).get("window_policy")
        if trained_window is not None and trained_window != asdict(policy):
            raise ValueError("runtime window differs from the frozen training window")
        self.head, self.analysis, self.policy = head, analysis, policy
        self._frames: deque[PoseFrame] = deque(maxlen=policy.max_buffer_frames)
        self._last: PoseFrame | None = None
        self._last_tick: datetime | None = None
        self._stream: tuple[str, int] | None = None
        self._camera_id: str | None = None
        self._ended_epoch = 0

    @property
    def buffer_size(self) -> int:
        return len(self._frames)

    def advance(self, frame: PoseFrame) -> FallBatch:
        stream = (frame.camera_id, frame.source_epoch)
        if self._camera_id is not None and frame.camera_id != self._camera_id:
            raise ValueError("one pipeline owns one camera; create a new pipeline for another camera")
        if frame.source_epoch <= self._ended_epoch:
            raise ValueError("a finalized source requires a newer source_epoch")
        if self._stream is not None and stream != self._stream:
            raise ValueError("finalize this source before changing camera or epoch")
        self._stream = stream
        self._camera_id = frame.camera_id
        previous = self._last
        if previous and (frame.sequence <= previous.sequence or frame.observed_at <= previous.observed_at):
            raise ValueError("pose frames must have strictly increasing sequence and timestamps")
        gap = previous and (frame.observed_at - previous.observed_at).total_seconds() > self.policy.max_frame_gap_seconds
        changed = previous and frame.track_key != previous.track_key
        reset_reason = "frame_gap" if gap else "track_change" if changed else None
        if not frame.valid or reset_reason:
            self._frames.clear()
        self._last = frame
        if frame.valid:
            self._frames.append(frame)
        cutoff = frame.observed_at - timedelta(seconds=self.policy.window_seconds)
        # Keep one predecessor of the left boundary for causal time-bin sampling.
        while len(self._frames) > 1 and self._frames[1].observed_at <= cutoff:
            self._frames.popleft()
        immediate = not frame.valid or reset_reason is not None
        if not immediate and self._last_tick is not None:
            if (frame.observed_at - self._last_tick).total_seconds() < self.policy.stride_seconds - 1e-9:
                return FallBatch(source_epoch=frame.source_epoch)
        self._last_tick = frame.observed_at
        score, quality = None, {}
        reason = frame.reason or "invalid_pose" if not frame.valid else reset_reason or "warming_up"
        if frame.valid and not reset_reason and self._frames[0].observed_at <= cutoff:
            samples = []
            candidates = tuple(self._frames)
            index = 0
            for number in range(self.policy.sample_count):
                target = cutoff + timedelta(seconds=(number + .5) * self.policy.window_seconds / self.policy.sample_count)
                while index + 1 < len(candidates) and candidates[index + 1].observed_at <= target:
                    index += 1
                samples.append(candidates[index])
            if len({item.sequence for item in samples}) != self.policy.sample_count:
                reason = "insufficient_sampling_density"
            else:
                score, quality = self.head.predict(tuple(samples))
                reason = "valid" if score is not None else "invalid_descriptor"
                quality = {**quality, "sample_sequences": [item.sequence for item in samples]}
        observation = FallObservation(
            camera_id=frame.camera_id, source_epoch=frame.source_epoch, sequence=frame.sequence,
            observed_at=frame.observed_at, window_started_at=max(cutoff, self._frames[0].observed_at)
            if self._frames else frame.observed_at, window_ended_at=frame.observed_at,
            score=score, model_version=self.head.model_version,
            subject_track_keys=(frame.track_key,) if frame.track_key else (), reason=reason,
        )
        records = self.analysis.advance(observation)
        return FallBatch(observation, tuple(records), quality, dict(self.analysis.last_record_reasons), frame.source_epoch)

    def finalize(self, reason: str = "source_end") -> FallBatch:
        if self._last is None:
            return FallBatch()
        frame = self._last
        records = self.analysis.finalize(frame.camera_id, frame.source_epoch, frame.observed_at, reason=reason)
        reasons = dict(self.analysis.last_record_reasons)
        self._ended_epoch = frame.source_epoch
        self._frames.clear()
        self._last = self._last_tick = self._stream = None
        return FallBatch(events=tuple(records), event_reasons=reasons, source_epoch=frame.source_epoch)
