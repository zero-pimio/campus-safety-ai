from campus_safety_ai.adapters.runtimes.tracker_factory import build_tracker
from campus_safety_ai.core.event_analysis import EventAnalysis
from campus_safety_ai.settings import load_intrusion_policy, load_video_runtime_settings


def default_analysis() -> EventAnalysis:
    runtime = load_video_runtime_settings()
    return EventAnalysis(load_intrusion_policy(), build_tracker(runtime.tracker_backend))
