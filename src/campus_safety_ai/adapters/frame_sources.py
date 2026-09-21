from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from campus_safety_ai.contracts import FramePacket, iso_time


class OpenCvFileFrames:
    """One finite local CFR replay; timestamps use frame index / file FPS.

    This adapter deliberately does not treat RTSP frame counts as a wall clock.
    It retains one frame and releases the decoder on EOF, errors or early exit.
    """

    def __init__(
        self, path: Path, camera_id: str, source_epoch: int, started_at: datetime,
        frame_stride: int = 1,
    ) -> None:
        if not path.is_file():
            raise ValueError("scene replay requires an existing local video file")
        if (
            type(frame_stride) is not int or type(source_epoch) is not int
            or frame_stride < 1 or source_epoch < 1 or not camera_id.strip()
        ):
            raise ValueError("frame_stride and source_epoch must be positive; camera_id must be non-empty")
        iso_time(started_at)
        import cv2

        self._capture = cv2.VideoCapture(str(path))
        self._closed = False
        self._iterated = False
        if not self._capture.isOpened():
            self.close()
            raise ValueError(f"cannot open video: {path}")
        try:
            self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            dimensions = [float(self._capture.get(key)) for key in (
                cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
            )]
            if any(not math.isfinite(value) or value <= 0 for value in (self.fps, *dimensions)):
                raise ValueError("video dimensions and FPS must be finite and positive")
            self.width, self.height = map(int, dimensions)
            count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.expected_frames = int(count) if math.isfinite(count) and count > 0 else None
        except BaseException:
            self.close()
            raise
        self.camera_id = camera_id
        self.source_epoch = source_epoch
        self.started_at = started_at
        self.frame_stride = frame_stride
        self.decoded_frames = 0

    def __iter__(self) -> Iterator[FramePacket]:
        if self._iterated:
            raise RuntimeError("a file source can only be consumed once")
        self._iterated = True
        try:
            while True:
                available, frame = self._capture.read()
                if not available:
                    if self.expected_frames is not None and self.decoded_frames < self.expected_frames:
                        raise ValueError(
                            f"video ended after {self.decoded_frames} of {self.expected_frames} declared frames"
                        )
                    break
                index = self.decoded_frames
                self.decoded_frames += 1
                if index % self.frame_stride:
                    continue
                height, width = frame.shape[:2]
                if (width, height) != (self.width, self.height):
                    raise ValueError("video dimensions changed within one source epoch")
                yield FramePacket(
                    camera_id=self.camera_id, source_epoch=self.source_epoch, sequence=index + 1,
                    captured_at=self.started_at + timedelta(seconds=index / self.fps),
                    received_at=datetime.now(UTC), width=width, height=height, image=frame,
                )
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> OpenCvFileFrames:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
