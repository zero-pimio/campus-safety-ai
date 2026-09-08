from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from campus_safety_ai.training.manifest import FightSample


class FightVideoDataset:
    def __new__(
        cls,
        project_root: Path,
        samples: list[FightSample],
        frame_count: int = 8,
        image_size: int = 224,
        training: bool = False,
    ) -> Any:
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as error:
            raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

        class _Dataset(Dataset):
            def __len__(self) -> int:
                return len(samples)

            def __getitem__(self, index: int) -> tuple[Any, int]:
                frames = _sample_frames(
                    project_root / samples[index].video_path,
                    frame_count,
                    image_size,
                    training,
                )
                return torch.from_numpy(frames), samples[index].label

        return _Dataset()


def _sample_frames(
    path: Path, frame_count: int, image_size: int, training: bool
) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"video has no frames: {path}")

    edges = np.linspace(0, total, frame_count + 1, dtype=np.int64)
    indices: list[int] = []
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        upper = max(int(left) + 1, int(right))
        index = random.randrange(int(left), upper) if training else (int(left) + upper - 1) // 2
        indices.append(min(index, total - 1))

    flip = training and random.random() < 0.5
    frames = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            available, frame = capture.read()
            if not available:
                raise ValueError(f"cannot decode frame {frame_index}: {path}")
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
                top = random.randint(0, height - image_size)
                left = random.randint(0, width - image_size)
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
