from __future__ import annotations

from typing import Any


class FightTsn:
    """Factory-compatible wrapper kept importable without the training extra."""

    def __new__(
        cls,
        frame_count: int = 8,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> Any:
        try:
            import torch
            from torch import nn
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as error:
            raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

        class _FightTsn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                backbone = mobilenet_v3_small(weights=weights)
                self.frame_count = frame_count
                self.features = backbone.features
                self.feature_dim = backbone.classifier[0].in_features
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(self.feature_dim, num_classes)

            def forward(self, frames: Any) -> Any:
                batch, channels, height, width = frames.shape
                images = frames.reshape(batch * self.frame_count, 3, height, width)
                features = self.features(images)
                features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
                features = features.reshape(batch, self.frame_count, self.feature_dim).mean(dim=1)
                return self.classifier(self.dropout(features))

        return _FightTsn()
