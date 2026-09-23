"""Finite video replay using decoded presentation timestamps, including VFR."""
from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from campus_safety_ai.contracts import FramePacket, iso_time


def video_timestamps(path: Path) -> list[float]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=120)
    raw = [float(row["best_effort_timestamp_time"]) for row in json.loads(result.stdout)["frames"]]
    if not raw or any(not math.isfinite(t) for t in raw):
        raise ValueError("decoded video timestamps are empty or nonfinite")
    times = [round(t - raw[0], 6) for t in raw]
    validate_timestamps(times)
    return times


def validate_timestamps(times: list[float]) -> None:
    if (not times or times[0] != 0 or any(not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:], strict=False))):
        raise ValueError("timestamps must start at zero and strictly increase")


class OpenCvTimedFileFrames:
    """Explicit sidecar times require caller verification against the source hash.

    Decode every frame and fail on count mismatch; never approximate with mean FPS.
    ffprobe scans the finite local file once. This class is not an RTSP adapter.
    """

    def __init__(self, path: Path, camera_id: str, source_epoch: int, started_at: datetime,
                 *, timestamps: list[float] | None = None):
        if not path.is_file() or not camera_id.strip() or type(source_epoch) is not int or source_epoch < 1:
            raise ValueError("local video, camera and positive source epoch are required")
        iso_time(started_at)
        self.timestamps = video_timestamps(path) if timestamps is None else list(timestamps)
        validate_timestamps(self.timestamps)
        import cv2
        self._capture = cv2.VideoCapture(str(path))
        self._closed = self._iterated = False
        try:
            if not self._capture.isOpened():
                raise ValueError("cannot open local video")
        except BaseException:
            self.close()
            raise
        self.camera_id, self.source_epoch, self.started_at = camera_id, source_epoch, started_at
        self.decoded_frames = 0

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("a file source can only be consumed once")
        self._iterated = True
        try:
            for index, seconds in enumerate(self.timestamps):
                if self._closed:
                    return
                ok, image = self._capture.read()
                if not ok:
                    raise ValueError("decoded frames fewer than presentation timestamps")
                self.decoded_frames += 1
                height, width = image.shape[:2]
                yield FramePacket(camera_id=self.camera_id, source_epoch=self.source_epoch, sequence=index + 1,
                                  captured_at=self.started_at + timedelta(seconds=seconds),
                                  received_at=datetime.now(UTC), width=width, height=height, image=image)
            if not self._closed and self._capture.read()[0]:
                raise ValueError("decoded frames exceed presentation timestamps")
        finally:
            self.close()

    def close(self):
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
