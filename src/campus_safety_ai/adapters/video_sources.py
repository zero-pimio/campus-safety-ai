from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from campus_safety_ai.core.fight_pipeline import VideoWindow


class OpenCvVideoSource:
    """OpenCV adapter for both finite MP4 files and RTSP URLs."""

    def __init__(self, uri: str | Path, frame_len: int = 8, sample_frequency: int = 7) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is not installed; install the fight-video extra") from error
        if frame_len <= 0 or sample_frequency <= 0:
            raise ValueError("frame_len and sample_frequency must be positive")

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(uri))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"cannot open video source: {uri}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            self._capture.release()
            raise ValueError("video source must report a positive FPS")
        if frame_count > 0 and frame_count < frame_len:
            self._capture.release()
            raise ValueError(f"video source must contain at least {frame_len} frames")
        self.sample_frequency = (
            min(sample_frequency, max(1, frame_count // frame_len))
            if frame_count > 0
            else sample_frequency
        )
        self.frame_len = frame_len
        self._closed = False

    def __iter__(self) -> Iterator[VideoWindow]:
        sampled: list[Any] = []
        sampled_indices: list[int] = []
        frame_index = 0
        try:
            while True:
                available, bgr_frame = self._capture.read()
                if not available:
                    break
                if frame_index % self.sample_frequency == 0:
                    sampled.append(self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB))
                    sampled_indices.append(frame_index)
                    if len(sampled) == self.frame_len:
                        yield VideoWindow(tuple(sampled), sampled_indices[0], sampled_indices[-1])
                        sampled.clear()
                        sampled_indices.clear()
                frame_index += 1
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "OpenCvVideoSource":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
