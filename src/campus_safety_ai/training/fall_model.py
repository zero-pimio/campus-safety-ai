"""Frozen image features with a trainable temporal summary classifier."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class FallTemporalClassifier(nn.Module):
    """Classify clips using mean, maximum, and last-minus-first frame features.

    Normalization must be fitted using the training split only. Keeping the
    backbone and normalization fixed makes cached-feature and full-clip forward
    passes use the same trainable linear head.
    """

    def __init__(self, num_classes: int = 2, frame_count: int = 16,
                 pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        for name, value in (("num_classes", num_classes), ("frame_count", frame_count)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(dropout) or dropout != 0:
            raise ValueError("dropout must be 0 for the deterministic cached-feature classifier")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        image_model = mobilenet_v3_small(weights=weights)
        self.frame_count = frame_count
        self.num_classes = num_classes
        self.feature_dim = int(image_model.classifier[0].in_features)
        self.summary_dim = 3 * self.feature_dim
        self.backbone = image_model.features
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("feature_mean", torch.zeros(self.summary_dim))
        self.register_buffer("feature_scale", torch.ones(self.summary_dim))
        self.head = nn.Linear(self.summary_dim, num_classes)

    def train(self, mode: bool = True) -> FallTemporalClassifier:
        super().train(mode)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        return self

    @staticmethod
    def _exporting() -> bool:
        return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()

    def forward_frame_features(self, frames: Tensor) -> Tensor:
        if not self._exporting() and (
            frames.ndim != 5 or frames.shape[1] != self.frame_count
            or frames.shape[2] != 3 or frames.shape[3] != 224 or frames.shape[4] != 224
        ):
            raise ValueError(f"frames must have shape [B, {self.frame_count}, 3, 224, 224]")
        batch = frames.shape[0]
        images = frames.reshape(batch * self.frame_count, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(images)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return features.reshape(batch, self.frame_count, self.feature_dim)

    def summarize_features(self, features: Tensor) -> Tensor:
        if not self._exporting() and (
            features.ndim != 3 or features.shape[1] != self.frame_count or features.shape[2] != self.feature_dim
        ):
            raise ValueError(f"features must have shape [B, {self.frame_count}, {self.feature_dim}]")
        return torch.cat((features.mean(dim=1), features.amax(dim=1), features[:, -1] - features[:, 0]), dim=1)

    def forward_features(self, features: Tensor) -> Tensor:
        summary = self.summarize_features(features)
        return self.head((summary - self.feature_mean) / self.feature_scale)

    def forward(self, frames: Tensor) -> Tensor:
        return self.forward_features(self.forward_frame_features(frames))

    def set_feature_normalization(self, mean: Tensor, scale: Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device)
        expected = (self.summary_dim,)
        if tuple(mean.shape) != expected or tuple(scale.shape) != expected:
            raise ValueError(f"normalization mean and scale must have shape {expected}")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("normalization mean must be finite and scale must be finite and positive")
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_scale.copy_(scale)
