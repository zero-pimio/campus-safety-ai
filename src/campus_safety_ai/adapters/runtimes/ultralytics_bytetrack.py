from __future__ import annotations

from argparse import Namespace
from typing import Any

from campus_safety_ai.contracts import BBox, Detections, Track
from campus_safety_ai.core.event_analysis import TrackingResult


class UltralyticsByteTracker:
    """Adapter from project detections to Ultralytics ByteTrack results."""

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as error:
            raise RuntimeError(
                "ByteTrack requires the vision extra, including lap>=0.5.12"
            ) from error

        self._tracker_type = BYTETracker
        self._args = Namespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._tracker: Any | None = None
        self._epoch: tuple[str, int] | None = None
        self._labels: dict[str, int] = {}
        self._reported_removed: set[int] = set()

    @staticmethod
    def _track_key(camera_id: str, source_epoch: int, local_id: int) -> str:
        return f"{camera_id}:{source_epoch}:bytetrack:{local_id}"

    def _reset(self, epoch: tuple[str, int]) -> None:
        self._tracker = self._tracker_type(self._args)
        self._epoch = epoch
        self._labels.clear()
        self._reported_removed.clear()

    def update(self, batch: Detections) -> TrackingResult:
        import numpy as np
        from ultralytics.engine.results import Boxes

        epoch = (batch.camera_id, batch.source_epoch)
        expired_keys: list[str] = []
        if epoch != self._epoch:
            if self._epoch is not None and self._tracker is not None:
                # Tracks cannot cross a source epoch (CONTEXT.md); report every
                # live or lost track so open events get closed instead of leaked.
                old_camera, old_epoch = self._epoch
                live_stracks = (
                    *getattr(self._tracker, "tracked_stracks", ()),
                    *getattr(self._tracker, "lost_stracks", ()),
                )
                expired_keys = [
                    self._track_key(old_camera, old_epoch, int(track.track_id))
                    for track in live_stracks
                ]
            self._reset(epoch)
        assert self._tracker is not None

        for detection in batch.detections:
            self._labels.setdefault(detection.label, len(self._labels))
        reverse_labels = {value: key for key, value in self._labels.items()}
        rows = np.asarray(
            [
                [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                    detection.confidence,
                    self._labels[detection.label],
                ]
                for detection in batch.detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        output = self._tracker.update(Boxes(rows, (batch.height, batch.width)))
        tracks = tuple(
            Track(
                track_key=self._track_key(batch.camera_id, batch.source_epoch, int(row[4])),
                label=reverse_labels[int(row[6])],
                confidence=float(row[5]),
                bbox=BBox(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                observed_at=batch.captured_at,
            )
            for row in output
        )
        removed_ids = {int(track.track_id) for track in self._tracker.removed_stracks}
        newly_removed = removed_ids - self._reported_removed
        self._reported_removed.update(newly_removed)
        expired_keys.extend(
            self._track_key(batch.camera_id, batch.source_epoch, track_id)
            for track_id in sorted(newly_removed)
        )
        return TrackingResult(tracks, tuple(dict.fromkeys(expired_keys)))
