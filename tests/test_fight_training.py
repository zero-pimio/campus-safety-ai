from __future__ import annotations

import csv
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from campus_safety_ai.training.manifest import build_fight_samples


class FightTrainingManifestTests(unittest.TestCase):
    def test_airtlab_paired_cameras_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label_dir in ("fight", "noFight"):
                target = root / "datasets/private/scfd" / label_dir
                target.mkdir(parents=True)
                for index in range(10):
                    (target / f"{index}.mp4").touch()

            airtlab = root / "datasets/private/airtlab/violence-detection-dataset"
            for label_dir in ("violent", "non-violent"):
                for camera in ("cam1", "cam2"):
                    target = airtlab / label_dir / camera
                    target.mkdir(parents=True)
                    for index in range(10):
                        (target / f"{index}.mp4").touch()
            for name, action in (
                ("violent-action-classes.csv", "fight"),
                ("nonviolent-action-classes.csv", "hug"),
            ):
                with (airtlab / name).open("w", newline="") as handle:
                    writer = csv.writer(handle, delimiter=";")
                    writer.writerow(["FILE", "ACTION CLASSES"])
                    for index in range(10):
                        writer.writerow([f"{index}.mp4", action])

            samples = build_fight_samples(root, seed=7)
            splits_by_group: dict[str, set[str]] = defaultdict(set)
            for sample in samples:
                splits_by_group[sample.group_id].add(sample.split)
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
            self.assertEqual({sample.split for sample in samples}, {"train", "val", "test"})

    def test_non_fight_weapon_only_airtlab_clips_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label_dir in ("fight", "noFight"):
                target = root / "datasets/private/scfd" / label_dir
                target.mkdir(parents=True)
                for index in range(4):
                    (target / f"{index}.mp4").touch()
            airtlab = root / "datasets/private/airtlab/violence-detection-dataset"
            for label_dir in ("violent", "non-violent"):
                for camera in ("cam1", "cam2"):
                    target = airtlab / label_dir / camera
                    target.mkdir(parents=True)
                    for index in range(6):
                        (target / f"{index}.mp4").touch()
            with (airtlab / "violent-action-classes.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["FILE", "ACTION CLASSES"])
                writer.writerows(
                    [
                        ["0.mp4", "gunshot"],
                        ["1.mp4", "stab"],
                        ["2.mp4", "fight"],
                        ["3.mp4", "punch"],
                        ["4.mp4", "kick"],
                        ["5.mp4", "slap"],
                    ]
                )
            with (airtlab / "nonviolent-action-classes.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["FILE", "ACTION CLASSES"])
                writer.writerows([[f"{index}.mp4", "hug"] for index in range(6)])

            samples = build_fight_samples(root, seed=7)
            violent_names = {
                Path(sample.video_path).name
                for sample in samples
                if sample.dataset == "airtlab" and sample.label == 1
            }
            self.assertEqual(violent_names, {"2.mp4", "3.mp4", "4.mp4", "5.mp4"})


if __name__ == "__main__":
    unittest.main()
