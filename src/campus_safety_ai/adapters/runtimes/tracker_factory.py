from __future__ import annotations

from campus_safety_ai.core.event_analysis import SimpleIoUTracker, Tracker


def build_tracker(backend: str) -> Tracker:
    if backend == "simple_iou":
        return SimpleIoUTracker()
    if backend == "bytetrack":
        from campus_safety_ai.adapters.runtimes.ultralytics_bytetrack import UltralyticsByteTracker

        return UltralyticsByteTracker()
    raise ValueError(f"unsupported tracker backend: {backend}")
