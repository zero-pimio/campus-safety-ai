from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def iso_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError("bboxXyxy must contain four numbers")
        return cls(*(float(value) for value in values))

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def bottom_center_normalized(self, width: int, height: int) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2 / width, self.y2 / height)


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    source_epoch: int
    sequence: int
    captured_at: datetime
    received_at: datetime
    width: int
    height: int
    format: str = "BGR"
    image: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        iso_time(self.captured_at)
        iso_time(self.received_at)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Detection":
        return cls(
            label=str(value["label"]),
            confidence=float(value["confidence"]),
            bbox=BBox.from_list(value["bboxXyxy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence, "bboxXyxy": self.bbox.to_list()}


@dataclass(frozen=True)
class Detections:
    camera_id: str
    source_epoch: int
    sequence: int
    captured_at: datetime
    width: int
    height: int
    detections: tuple[Detection, ...]
    model_version: str
    inference_ms: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Detections":
        return cls(
            camera_id=str(value["cameraId"]),
            source_epoch=int(value["sourceEpoch"]),
            sequence=int(value["sequence"]),
            captured_at=parse_time(value["capturedAt"]),
            width=int(value["width"]),
            height=int(value["height"]),
            detections=tuple(Detection.from_dict(item) for item in value.get("detections", [])),
            model_version=str(value["modelVersion"]),
            inference_ms=float(value.get("inferenceMs", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameraId": self.camera_id,
            "sourceEpoch": self.source_epoch,
            "sequence": self.sequence,
            "capturedAt": iso_time(self.captured_at),
            "width": self.width,
            "height": self.height,
            "detections": [item.to_dict() for item in self.detections],
            "modelVersion": self.model_version,
            "inferenceMs": self.inference_ms,
        }


@dataclass(frozen=True)
class Track:
    track_key: str
    label: str
    confidence: float
    bbox: BBox
    observed_at: datetime


@dataclass(frozen=True)
class BehaviorObservation:
    """A model score for one behavior over a bounded video window."""

    camera_id: str
    source_epoch: int
    sequence: int
    observed_at: datetime
    window_started_at: datetime
    window_ended_at: datetime
    behavior: str
    score: float
    model_version: str
    subject_track_keys: tuple[str, ...] = ()
    evidence_uris: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        iso_time(self.observed_at)
        iso_time(self.window_started_at)
        iso_time(self.window_ended_at)
        if self.window_ended_at < self.window_started_at:
            raise ValueError("behavior window must end after it starts")
        if self.observed_at < self.window_ended_at:
            raise ValueError("observation cannot be available before its video window ends")
        if not 0 <= self.score <= 1:
            raise ValueError("behavior score must be between 0 and 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorObservation":
        return cls(
            camera_id=str(value["cameraId"]),
            source_epoch=int(value["sourceEpoch"]),
            sequence=int(value["sequence"]),
            observed_at=parse_time(value["observedAt"]),
            window_started_at=parse_time(value["windowStartedAt"]),
            window_ended_at=parse_time(value["windowEndedAt"]),
            behavior=str(value["behavior"]),
            score=float(value["score"]),
            model_version=str(value["modelVersion"]),
            subject_track_keys=tuple(str(key) for key in value.get("subjectTrackKeys", [])),
            evidence_uris=tuple(str(uri) for uri in value.get("evidenceUris", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameraId": self.camera_id,
            "sourceEpoch": self.source_epoch,
            "sequence": self.sequence,
            "observedAt": iso_time(self.observed_at),
            "windowStartedAt": iso_time(self.window_started_at),
            "windowEndedAt": iso_time(self.window_ended_at),
            "behavior": self.behavior,
            "score": self.score,
            "modelVersion": self.model_version,
            "subjectTrackKeys": list(self.subject_track_keys),
            "evidenceUris": list(self.evidence_uris),
        }


EventPhase = Literal["START", "UPDATE", "END"]


@dataclass(frozen=True)
class EventRecord:
    schema_version: str
    event_id: str
    revision: int
    phase: EventPhase
    event_type: str
    severity: str
    edge_id: str
    camera_id: str
    started_at: datetime
    observed_at: datetime
    ended_at: datetime | None
    subject_track_keys: tuple[str, ...]
    confidence: float
    model_version: str
    config_version: str
    idempotency_key: str
    status: str
    evidence_uris: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            "schemaVersion": value["schema_version"],
            "eventId": value["event_id"],
            "revision": value["revision"],
            "phase": value["phase"],
            "eventType": value["event_type"],
            "severity": value["severity"],
            "edgeId": value["edge_id"],
            "cameraId": value["camera_id"],
            "startedAt": iso_time(self.started_at),
            "observedAt": iso_time(self.observed_at),
            "endedAt": iso_time(self.ended_at) if self.ended_at else None,
            "subjectTrackKeys": list(self.subject_track_keys),
            "confidence": value["confidence"],
            "modelVersion": value["model_version"],
            "configVersion": value["config_version"],
            "idempotencyKey": value["idempotency_key"],
            "status": value["status"],
            "evidenceUris": list(self.evidence_uris),
        }
