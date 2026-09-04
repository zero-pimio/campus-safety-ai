import tempfile
import unittest
from pathlib import Path

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

        self.assertEqual(runtime.adapter, "opencv")
        self.assertEqual(runtime.tracker_backend, "simple_iou")
        self.assertEqual((runtime.frame_count, runtime.sample_frequency), (8, 7))
        self.assertEqual(model.backend, "paddle")
        self.assertEqual(model.path, PROJECT_ROOT / "models/ppTSM")
        self.assertEqual(platform.destination, "jsonl")
        self.assertEqual(fight.start_score, 0.75)
        self.assertEqual(intrusion.zone_id, "gate-02-restricted")

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
