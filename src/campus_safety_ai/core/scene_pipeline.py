from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from campus_safety_ai.contracts import Detections, EventRecord, FramePacket
from campus_safety_ai.core.perception import Perception


class SceneAnalysis(Protocol):
    def advance(self, batch: Detections) -> list[EventRecord]: ...

    def finalize(self, camera_id: str, source_epoch: int, ended_at: datetime) -> list[EventRecord]: ...


@dataclass(frozen=True)
class SceneFrameResult:
    frame: FramePacket | None
    detections: Detections | None
    events: tuple[EventRecord, ...]
    finish_reason: str | None = None


class SceneVideoPipeline:
    """Run interchangeable scene rules over real perception or fixed detections.

    The caller owns delivery and source cleanup. Full consumption emits source
    closure records at EOF. Decoder/perception errors first emit a closure at
    the last accepted observation, then re-raise; they are never normal EOF.
    """

    def __init__(self, perception: Perception, analysis: SceneAnalysis) -> None:
        self.perception = perception
        self.analysis = analysis

    def stream(self, frames: Iterable[FramePacket]) -> Iterator[SceneFrameResult]:
        last: FramePacket | None = None
        try:
            for frame in frames:
                batch = self.perception.detect(frame)
                if (
                    batch.camera_id, batch.source_epoch, batch.sequence, batch.captured_at,
                    batch.width, batch.height,
                ) != (
                    frame.camera_id, frame.source_epoch, frame.sequence, frame.captured_at,
                    frame.width, frame.height,
                ):
                    raise ValueError("perception changed source frame metadata")
                records = self.analysis.advance(batch)
                last = frame
                yield SceneFrameResult(frame, batch, tuple(records))
        except Exception:
            if last is not None:
                records = self.analysis.finalize(last.camera_id, last.source_epoch, last.captured_at)
                yield SceneFrameResult(None, None, tuple(records), "error")
            raise
        if last is not None:
            records = self.analysis.finalize(last.camera_id, last.source_epoch, last.captured_at)
            yield SceneFrameResult(None, None, tuple(records), "eof")
