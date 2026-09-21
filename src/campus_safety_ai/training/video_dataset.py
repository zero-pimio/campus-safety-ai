from __future__ import annotations

import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from campus_safety_ai.training.manifest import FightSample


class FightVideoDataset:
    """Pickleable map-style dataset; optional training imports stay lazy."""

    def __init__(
        self,
        project_root: Path,
        samples: list[FightSample],
        frame_count: int = 8,
        image_size: int = 224,
        training: bool = False,
    ) -> None:
        _validate_dimensions(frame_count, image_size)
        self.project_root = Path(project_root)
        self.samples = tuple(samples)
        self.frame_count = frame_count
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the training extra: pip install -e '.[training]'") from error
        sample = self.samples[index]
        frames = _sample_frames(
            self.project_root / sample.video_path,
            self.frame_count,
            self.image_size,
            self.training,
        )
        return torch.from_numpy(frames), sample.label


def _validate_dimensions(frame_count: int, image_size: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (frame_count, image_size)):
        raise ValueError("frame_count and image_size must be positive integers")


def _read_sampled_frames(capture: Any, indices: list[int], path: Path) -> Iterator[Any]:
    import cv2

    next_index = 0
    previous_index = None
    previous_frame = None
    for frame_index in indices:
        if frame_index == previous_index:
            yield previous_frame.copy()
            continue
        # Nearby targets avoid repeated keyframe seeks; distant targets keep
        # bounded traversal so long recordings do not require full decoding.
        distance = frame_index - previous_index if previous_index is not None else frame_index
        if frame_index < next_index or distance > 128:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"cannot seek to frame {frame_index}: {path}")
            next_index = frame_index
        while next_index < frame_index:
            if not capture.grab():
                raise ValueError(f"cannot decode frame {next_index}: {path}")
            next_index += 1
        available, frame = capture.read()
        if not available:
            raise ValueError(f"cannot decode frame {frame_index}: {path}")
        next_index = frame_index + 1
        previous_index, previous_frame = frame_index, frame
        yield frame


def _sample_frames(
    path: Path, frame_count: int, image_size: int, training: bool
) -> Any:
    _validate_dimensions(frame_count, image_size)
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        total_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(total_value) or total_value < 1:
            raise ValueError(f"video must report a positive finite frame count: {path}")
        total = int(total_value)

        edges = np.linspace(0, total, frame_count + 1, dtype=np.int64)
        indices: list[int] = []
        for left, right in zip(edges[:-1], edges[1:], strict=False):
            upper = max(int(left) + 1, int(right))
            index = random.randrange(int(left), upper) if training else (int(left) + upper - 1) // 2
            indices.append(min(index, total - 1))

        # Share spatial augmentation across the clip to avoid artificial motion.
        flip = training and random.random() < 0.5
        crop_top = random.random() if training else 0.5
        crop_left = random.random() if training else 0.5
        for frame in _read_sampled_frames(capture, indices, path):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            scale = (image_size + 32) / min(height, width)
            resized = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            height, width = resized.shape[:2]
            if training:
                top = round(crop_top * (height - image_size))
                left = round(crop_left * (width - image_size))
            else:
                top = (height - image_size) // 2
                left = (width - image_size) // 2
            crop = resized[top : top + image_size, left : left + image_size]
            if flip:
                crop = np.ascontiguousarray(crop[:, ::-1])
            frames.append(crop.transpose(2, 0, 1))
    finally:
        capture.release()

    tensor = np.concatenate(frames, axis=0).astype(np.float32) / 255.0
    mean = np.tile(np.array([0.485, 0.456, 0.406], dtype=np.float32), frame_count)
    std = np.tile(np.array([0.229, 0.224, 0.225], dtype=np.float32), frame_count)
    return (tensor - mean[:, None, None]) / std[:, None, None]
