from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class FightPrediction:
    """Two-class model output normalized to the project's fight score."""

    logits: tuple[float, float]
    fight_score: float


class FightClassifier(Protocol):
    """Runtime seam shared by Paddle, PyTorch, ONNX, and test adapters."""

    model_version: str
    frame_len: int

    def predict(self, rgb_frames: Sequence[Any]) -> FightPrediction: ...
