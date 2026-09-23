"""Pose-cache replay and receive-clock RTSP frames for the fall pipeline."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, UTC
from pathlib import Path
from threading import Event

from campus_safety_ai.adapters.runtimes.pose_fall import sha256
from campus_safety_ai.contracts import FramePacket
from campus_safety_ai.core.fall_pipeline import PoseFrame


def cached_pose_frames(path: Path, head, camera_id: str, source_epoch: int, started_at: datetime):
    identity_path = path.parent / "cache-identity.json"
    identity = json.loads(identity_path.read_text())
    prepared = json.loads((path.parent / "prepared.json").read_text())
    record = json.loads(path.read_text())
    if (prepared["cache_identity_sha256"] != sha256(identity_path)
            or record["cache_identity_sha256"] != sha256(identity_path)
            or prepared["record_sha256"].get(record["sequence_id"]) != sha256(path)):
        raise ValueError("pose cache identity or record fingerprint mismatch")
    for key in ("pose_checkpoint_sha256", "settings", "source_sha256"):
        if identity[key] != head.extraction_identity[key]:
            raise ValueError(f"pose cache differs from training: {key}")
    fields = ("timestamps_ms", "keypoints", "boxes", "tracking")
    rows = [record[key] for key in fields]
    if not rows[0] or len({len(row) for row in rows}) != 1:
        raise ValueError("pose cache must have aligned nonempty frames")
    offset = rows[0][0]
    generation, previous_track = 0, None
    for index, (timestamp, points, box, tracking) in enumerate(zip(*rows, strict=True), start=1):
        count = tracking["detection_count"]
        valid = tracking["association_valid"] is True and count == 1
        track = tracking["track_id"]
        reason = tracking["status"]
        if count != 1:
            reason = "multiple_people" if count > 1 else "missing_person"
        if not valid:
            previous_track = None
            key = None
        else:
            if track != previous_track:
                generation += 1
            previous_track = track
            key = f"{camera_id}:{source_epoch}:pose-{generation}"
        yield PoseFrame(camera_id, source_epoch, index,
                        started_at + timedelta(milliseconds=timestamp - offset), points, box,
                        key, valid, reason)


class RtspSourceError(OSError):
    """Credential-free transport failure; applications may reconnect only these."""


class RtspOpenError(RtspSourceError):
    pass


class RtspReadError(RtspSourceError):
    pass


def _open_rtsp_capture(uri: str, open_timeout_ms: int, read_timeout_ms: int):
    import cv2

    return cv2.VideoCapture(uri, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
                                               cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms])


class OpenCvRtspFrames:
    """One RTSP epoch. The application owns retry and event finalization.

    Receive timestamps use a monotonic clock anchored to UTC, not frame count/FPS.
    They do not measure capture/network latency. Open/read calls have backend
    timeouts; stop_event interrupts between reads, and close releases the capture.
    """

    def __init__(self, uri: str, camera_id: str, source_epoch: int, *, stop_event: Event | None = None,
                 open_timeout_ms: int = 5000, read_timeout_ms: int = 5000,
                 capture_factory: Callable | None = None, monotonic: Callable = time.monotonic,
                 utcnow: Callable = lambda: datetime.now(UTC),
                 timestamp_anchor: tuple[datetime, float] | None = None):
        if not uri.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("RTSP source must use rtsp:// or rtsps://")
        if not camera_id.strip() or type(source_epoch) is not int or source_epoch < 1:
            raise ValueError("camera and positive source_epoch are required")
        if any(type(value) is not int or value < 1 for value in (open_timeout_ms, read_timeout_ms)):
            raise ValueError("RTSP open and read timeouts must be positive integer milliseconds")
        self.camera_id, self.source_epoch = camera_id, source_epoch
        self._stop = stop_event or Event()
        self._monotonic, self._utcnow = monotonic, utcnow
        self._timestamp_anchor = timestamp_anchor
        self._capture = None
        self._closed = False
        self._iterated = False
        try:
            self._capture = (capture_factory or _open_rtsp_capture)(uri, open_timeout_ms, read_timeout_ms)
            if not self._capture.isOpened():
                raise RtspOpenError("cannot open RTSP source")
        except BaseException as error:
            self.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RtspOpenError("cannot open RTSP source") from None

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("an RTSP source epoch can only be consumed once")
        self._iterated = True
        anchor, tick = self._timestamp_anchor or (self._utcnow(), self._monotonic())
        sequence = 0
        try:
            while not self._closed and not self._stop.is_set():
                try:
                    available, image = self._capture.read()
                except Exception:
                    raise RtspReadError("RTSP read failed; source epoch closed") from None
                if not available:
                    if self._stop.is_set() or self._closed:
                        break
                    raise RtspReadError("RTSP read ended or timed out; source epoch closed")
                if self._stop.is_set() or self._closed:
                    break
                sequence += 1
                captured_at = anchor + timedelta(seconds=self._monotonic() - tick)
                try:
                    height, width = image.shape[:2]
                    if height < 1 or width < 1:
                        raise ValueError("empty frame")
                except Exception:
                    raise RtspReadError("RTSP decoder returned an invalid frame") from None
                yield FramePacket(self.camera_id, self.source_epoch, sequence, captured_at, self._utcnow(),
                                  width, height, image=image)
        finally:
            self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            if self._capture is not None:
                # Never mask the original failure or expose a backend exception
                # containing the input URI while unwinding another error.
                try:
                    self._capture.release()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
