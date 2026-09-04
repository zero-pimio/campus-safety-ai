from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from campus_safety_ai.contracts import BehaviorObservation


class NullEvidenceSink:
    def capture(
        self, observation: BehaviorObservation, rgb_frames: Sequence[Any], sampled_fps: float
    ) -> dict[str, str] | None:
        return None


class DirectoryEvidenceSink:
    """Local evidence adapter that writes a snapshot, sampled clip, and manifest."""

    def __init__(self, directory: Path, threshold: float) -> None:
        self.directory = directory
        self.threshold = threshold
        directory.mkdir(parents=True, exist_ok=True)
        self.manifest = directory / "manifest.jsonl"

    def capture(
        self, observation: BehaviorObservation, rgb_frames: Sequence[Any], sampled_fps: float
    ) -> dict[str, str] | None:
        if observation.score < self.threshold or not rgb_frames:
            return None
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("install the vision extra to write video evidence") from error

        camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", observation.camera_id).strip("._") or "camera"
        stem = f"{camera}-e{observation.source_epoch}-s{observation.sequence}"
        snapshot = self.directory / f"{stem}.jpg"
        clip = self.directory / f"{stem}.mp4"
        first = np.asarray(rgb_frames[0])
        height, width = first.shape[:2]
        if not cv2.imwrite(str(snapshot), cv2.cvtColor(first, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"cannot create evidence snapshot: {snapshot}")
        writer = cv2.VideoWriter(
            str(clip), cv2.VideoWriter_fourcc(*"mp4v"), max(sampled_fps, 1.0), (width, height)
        )
        if not writer.isOpened():
            raise ValueError(f"cannot create evidence clip: {clip}")
        try:
            for frame in rgb_frames:
                writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        artifact = {"snapshot": str(snapshot), "clip": str(clip)}
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"observation": observation.to_dict(), "artifact": artifact},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        return artifact
