from pathlib import Path

from campus_safety_ai.adapters.runtimes.tracker_factory import build_tracker
from campus_safety_ai.core.event_analysis import EventAnalysis
from campus_safety_ai.settings import PROJECT_ROOT, load_intrusion_policy, load_video_runtime_settings


def default_analysis(
    event_config: Path = PROJECT_ROOT / "configs/events/intrusion-v1.toml",
    scene_config: Path = PROJECT_ROOT / "configs/scenes/gate-02.toml",
    tracker_backend: str | None = None,
) -> EventAnalysis:
    runtime = load_video_runtime_settings()
    return EventAnalysis(
        load_intrusion_policy(event_config, scene_config),
        build_tracker(tracker_backend or runtime.tracker_backend),
    )
