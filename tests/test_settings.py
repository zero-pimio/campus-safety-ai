import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from campus_safety_ai.contracts import BehaviorObservation
from campus_safety_ai.core.fight_analysis import FightEventAnalysis
from campus_safety_ai.settings import (
    PROJECT_ROOT,
    load_fight_model_settings,
    load_fight_policy,
    load_intrusion_policy,
    load_platform_settings,
    load_video_runtime_settings,
)


class SettingsTests(unittest.TestCase):
    def test_repository_configs_drive_runtime_values(self) -> None:
        runtime = load_video_runtime_settings()
        model = load_fight_model_settings(runtime.fight_model_config)
        platform = load_platform_settings(runtime.platform_config)
        fight = load_fight_policy(runtime.fight_event_config)
        intrusion = load_intrusion_policy()

        self.assertEqual(runtime.video_adapter, "opencv")
        self.assertEqual(runtime.perception_adapter, "ultralytics")
        self.assertEqual(runtime.tracker_backend, "simple_iou")
        self.assertEqual((runtime.frame_count, runtime.sample_frequency), (8, 7))
        self.assertEqual(model.backend, "paddle")
        self.assertEqual(model.path, PROJECT_ROOT / "models/ppTSM")
        self.assertEqual(platform.destination, "jsonl")
        self.assertEqual(fight.start_score, 0.75)
        self.assertEqual(intrusion.zone_id, "gate-02-restricted")

    def test_event_type_and_behavior_label_are_wired_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fight.toml"
            path.write_text(
                'schema_version = "1.0"\n'
                'config_version = "fight-test"\n'
                "start_score = 0.8\n"
                "end_score = 0.3\n"
                "confirm_seconds = 1.0\n"
                "clear_seconds = 1.0\n"
                "cooldown_seconds = 5.0\n"
                'behavior_label = "brawl"\n'
                'event_type = "brawl_detected"\n',
                encoding="utf-8",
            )
            policy = load_fight_policy(path)
            self.assertEqual(policy.behavior_label, "brawl")
            self.assertEqual(policy.event_type, "brawl_detected")

            observed = datetime(2026, 8, 25, tzinfo=UTC)
            analysis = FightEventAnalysis(policy)
            started = analysis.advance(
                BehaviorObservation(
                    camera_id="gate-02",
                    source_epoch=1,
                    sequence=1,
                    observed_at=observed,
                    window_started_at=observed - timedelta(seconds=1),
                    window_ended_at=observed,
                    behavior="brawl",
                    score=0.95,
                    model_version="test-v1",
                )
            )
            self.assertEqual([record.phase for record in started], ["START"])
            self.assertEqual(started[0].event_type, "brawl_detected")

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.toml"
            path.write_text(
                'schema_version = "2.0"\nbackend = "torch"\npath = "x"\nmodel_version = "x"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_fight_model_settings(path)


if __name__ == "__main__":
    unittest.main()
