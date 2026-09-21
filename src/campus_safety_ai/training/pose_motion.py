"""Dynamic-only pose descriptors and a small learned motion classifier.

Descriptors use observed adjacent pairs only. No RGB, absolute position, static
pose value, or raw confidence is included in the feature vector. Invalid clips
return NaN features and must be excluded before fitting or inference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

CORE_INDICES = (5, 6, 11, 12)  # left/right shoulder, left/right hip in COCO-17
MAX_PAIR_GAP_SECONDS = 2.0
_SIGNED_STATS = ("max_positive_rate", "max_negative_rate", "mean_abs_rate", "net_change", "total_path", "rate_std")
FEATURE_NAMES = tuple(f"{channel}_{stat}" for channel in (
    "hip_vertical", "hip_horizontal", "torso_angle", "log_bbox_aspect",
) for stat in _SIGNED_STATS) + (
    "core_shape_max_rate", "core_shape_mean_rate", "core_shape_total_path", "core_shape_rate_std",
    "shoulder_width_max_abs_rate", "shoulder_width_total_path",
    "hip_width_max_abs_rate", "hip_width_total_path",
)


def _signed_statistics(changes: np.ndarray, seconds: np.ndarray) -> list[float]:
    rates = changes / seconds
    return [max(0.0, float(rates.max())), max(0.0, float((-rates).max())), float(np.abs(rates).mean()),
            float(changes.sum()), float(np.abs(changes).sum()), float(rates.std())]


def motion_descriptor(keypoints: np.ndarray, boxes: np.ndarray, timestamps_ms: Any,
                      min_conf: float = 0.25) -> dict[str, Any]:
    """Return 32 explainable motion features, feature names, and quality.

    Coordinates are in one common image coordinate system; boxes are XYXY.
    All four core joints must be finite and meet min_conf. Box area and core
    geometry must be nondegenerate. Positive hip_vertical means downward in
    image coordinates; positive torso_angle turns the shoulder-to-hip axis
    clockwise. Angular changes use wrapped radians. Translation is divided by
    the mean sqrt(box area) of each adjacent pair, and rates use actual seconds.

    A global image translation or positive uniform scaling leaves the descriptor
    unchanged. This does not compensate for camera motion during a clip. At
    least four valid frames and three valid original adjacent pairs are required.
    Adjacent observations over two seconds apart are never differenced.
    """
    if isinstance(min_conf, bool) or not math.isfinite(min_conf) or not 0 <= min_conf <= 1:
        raise ValueError("min_conf must be finite and in [0, 1]")
    keypoints = np.asarray(keypoints, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    times = np.asarray(timestamps_ms, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError("keypoints must have shape [T, 17, 3]")
    count = keypoints.shape[0]
    if boxes.shape != (count, 4) or times.shape != (count,):
        raise ValueError("boxes [T, 4] and timestamps_ms [T] must align with keypoints")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps_ms must be finite and strictly increasing")
    seconds = np.diff(times) / 1000
    if not np.isfinite(seconds).all() or np.any(seconds <= 0):
        raise ValueError("timestamp deltas in seconds must be finite and positive")
    core = keypoints[:, CORE_INDICES, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        scale = np.sqrt(width * height)
        shoulders = core[:, :2, :2].mean(axis=1)
        hips = core[:, 2:, :2].mean(axis=1)
        torso = shoulders - hips
        shoulder_width = np.linalg.norm(core[:, 1, :2] - core[:, 0, :2], axis=1)
        hip_width = np.linalg.norm(core[:, 3, :2] - core[:, 2, :2], axis=1)
        torso_length = np.linalg.norm(torso, axis=1)
        valid_frames = (
            np.isfinite(core).all(axis=(1, 2))
            & (core[:, :, 2] >= min_conf).all(axis=1) & (core[:, :, 2] <= 1).all(axis=1)
            & np.isfinite(boxes).all(axis=1) & np.isfinite(scale) & (width > 0) & (height > 0) & (scale > 0)
            & np.isfinite(torso_length) & (torso_length > 0)
            & np.isfinite(shoulder_width) & (shoulder_width > 0)
            & np.isfinite(hip_width) & (hip_width > 0)
        )
    valid_pairs = valid_frames[:-1] & valid_frames[1:] & (seconds <= MAX_PAIR_GAP_SECONDS)
    frame_count, pair_count = int(valid_frames.sum()), int(valid_pairs.sum())
    reasons = []
    if frame_count < 4:
        reasons.append("fewer_than_4_valid_frames")
    if pair_count < 3:
        reasons.append("fewer_than_3_valid_adjacent_pairs")
    quality = {
        "frame_count": count, "valid_frame_count": frame_count, "valid_pair_count": pair_count,
        "valid_frame_fraction": frame_count / count if count else 0.0,
        "valid_pair_fraction": pair_count / (count - 1) if count > 1 else 0.0,
        "valid_frame_mask": valid_frames.tolist(), "valid_pair_mask": valid_pairs.tolist(),
        "core_keypoint_indices": list(CORE_INDICES), "minimum_confidence": float(min_conf),
        "maximum_pair_gap_seconds": MAX_PAIR_GAP_SECONDS,
        "long_gap_pair_count": int((seconds > MAX_PAIR_GAP_SECONDS).sum()),
        "valid_pair_duration_seconds": float(seconds[valid_pairs].sum()), "invalid_reasons": reasons,
    }
    result = {"features": np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32),
              "feature_names": list(FEATURE_NAMES), "quality": quality, "valid": False}
    if reasons:
        return result
    indices = np.flatnonzero(valid_pairs)
    elapsed = seconds[indices]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pair_scale = (scale[indices] + scale[indices + 1]) / 2
        hip_changes = (hips[indices + 1] - hips[indices]) / pair_scale[:, None]
        angles = np.arctan2(torso[:, 0], -torso[:, 1])
        angle_changes = angles[indices + 1] - angles[indices]
        angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))
        log_aspect = np.log(width / height)
        aspect_changes = log_aspect[indices + 1] - log_aspect[indices]
        features = []
        for changes in (hip_changes[:, 1], hip_changes[:, 0], angle_changes, aspect_changes):
            features.extend(_signed_statistics(changes, elapsed))
        normalized_core = (core[:, :, :2] - hips[:, None, :]) / scale[:, None, None]
        core_changes = np.linalg.norm(normalized_core[indices + 1] - normalized_core[indices], axis=2).mean(axis=1)
        core_rates = core_changes / elapsed
        features.extend((float(core_rates.max()), float(core_rates.mean()),
                         float(core_changes.sum()), float(core_rates.std())))
        for normalized_width in (shoulder_width / scale, hip_width / scale):
            changes = normalized_width[indices + 1] - normalized_width[indices]
            features.extend((float((np.abs(changes) / elapsed).max()), float(np.abs(changes).sum())))
        values = np.asarray(features, dtype=np.float32)
    if not np.isfinite(values).all():
        quality["invalid_reasons"].append("nonfinite_descriptor")
        return result
    result.update(features=values, valid=True)
    return result


class PoseMotionClassifier(nn.Module):
    """Train-only normalization followed by a learned linear or 16-unit head."""

    def __init__(self, feature_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim not in (0, 16):
            raise ValueError("hidden_dim must be 0 or 16")
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.head = (nn.Linear(feature_dim, 2) if hidden_dim == 0 else nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2),
        ))

    def set_feature_normalization(self, mean: Tensor, scale: Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device)
        if tuple(mean.shape) != (self.feature_dim,) or tuple(scale.shape) != (self.feature_dim,):
            raise ValueError(f"normalization mean and scale must have shape [{self.feature_dim}]")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("normalization mean must be finite and scale must be finite and positive")
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_scale.copy_(scale)

    def forward(self, features: Tensor) -> Tensor:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if features.ndim != 2 or features.shape[1] != self.feature_dim:
                raise ValueError(f"features must have shape [B, {self.feature_dim}]")
        return self.head((features - self.feature_mean) / self.feature_scale)
