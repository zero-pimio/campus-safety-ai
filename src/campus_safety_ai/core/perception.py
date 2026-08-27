from __future__ import annotations

from typing import Protocol

from campus_safety_ai.contracts import Detections, FramePacket


class Perception(Protocol):
    def detect(self, frame: FramePacket) -> Detections:
        """Return project-native detections; never leak vendor result types."""

