from campus_safety_ai.core.event_analysis import EventAnalysis, IntrusionPolicy


def default_analysis() -> EventAnalysis:
    return EventAnalysis(
        IntrusionPolicy(
            edge_id="edge-dev-01",
            zone_id="gate-02-restricted",
            polygon=((0.45, 0.2), (0.95, 0.2), (0.95, 0.95), (0.45, 0.95)),
            enter_seconds=2.0,
            exit_seconds=1.0,
            cooldown_seconds=10.0,
            config_version="gate-02-intrusion-v1",
        )
    )

