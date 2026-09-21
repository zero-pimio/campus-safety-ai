from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campus_safety_ai.apps.evaluate_fight import _sha256, evaluate
from campus_safety_ai.training.calibration import (
    predictions_at_threshold,
    select_validation_threshold,
    threshold_from_validation_report,
)
from campus_safety_ai.training.manifest import FightSample


class ThresholdSelectionTests(unittest.TestCase):
    def test_default_and_boundary_rule(self) -> None:
        self.assertEqual(predictions_at_threshold([0.1, 0.5, 0.9]), [0, 1, 1])
        self.assertEqual(predictions_at_threshold([0.5, 0.8], 0.8), [0, 1])

    def test_validation_selection_separates_scores_below_default(self) -> None:
        selection = select_validation_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], split="val")
        self.assertAlmostEqual(selection["threshold"], 0.25)
        self.assertEqual(selection["metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(selection["objective"], "balanced_accuracy")

    def test_equivalent_candidates_prefer_default(self) -> None:
        selection = select_validation_threshold([0, 1], [0.2, 0.7], split="val")
        self.assertEqual(selection["threshold"], 0.5)

    def test_selection_requires_complete_val_and_both_classes(self) -> None:
        for parameters in ({"split": "test"}, {"split": "train"}, {"split": "val", "partial": True}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                select_validation_threshold([0, 1], [0.2, 0.7], **parameters)
        for labels, scores in (([0], [0.1]), ([0, 1], [0.2]), ([], []), ([0, 2], [0.1, 0.8])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                select_validation_threshold(labels, scores, split="val")

    def test_invalid_probabilities_are_rejected(self) -> None:
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predictions_at_threshold([0.2], value)
            with self.subTest(score=value), self.assertRaises(ValueError):
                predictions_at_threshold([value])

    def test_frozen_report_rejects_wrong_provenance_and_partial_results(self) -> None:
        report = {
            "split": "val", "partial": False, "sample_count": 2, "available_sample_count": 2,
            "predictions": [{}, {}], "checkpoint_sha256": "model", "manifest_sha256": "manifest",
            "threshold": 0.4,
        }
        self.assertEqual(threshold_from_validation_report(
            report, checkpoint_sha256="model", manifest_sha256="manifest"
        ), 0.4)
        for changes in (
            {"split": "test"}, {"partial": True}, {"sample_count": 1}, {"predictions": []}, {"predictions": None},
            {"checkpoint_sha256": "other"}, {"manifest_sha256": "other"}, {"threshold": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                threshold_from_validation_report(
                    {**report, **changes}, checkpoint_sha256="model", manifest_sha256="manifest"
                )


class EvaluationPreflightTests(unittest.TestCase):
    def test_invalid_arguments_fail_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for parameters in (
                {"split": "invalid"}, {"batch_size": 0}, {"batch_size": 1.5}, {"workers": -1},
                {"max_samples": 0}, {"max_samples": -2}, {"max_samples": True}, {"threshold": float("nan")},
                {"select_threshold": True}, {"threshold": 0.5, "select_threshold": True},
                {"threshold": 0.5, "threshold_report": root / "val.json"},
            ):
                args = {
                    "project_root": root, "manifest_path": root / "missing.csv",
                    "checkpoint_path": root / "missing.pt", "split": "test", "output_path": root / "out.json",
                }
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    evaluate(**{**args, **parameters})

    def test_existing_output_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out.json"
            output.write_text("existing result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                evaluate(root, root / "missing.csv", root / "missing.pt", "val", output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing result")


class EvaluationReportTests(unittest.TestCase):
    def test_reports_complete_predictions_datasets_and_frozen_test_threshold(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional training extra is not installed")

        class Model:
            def load_state_dict(self, state):
                pass

            def eval(self):
                return self

            def to(self, device):
                return self

            def __call__(self, frames):
                return frames

        def dataset(root, samples, frame_count, image_size, training):
            scores = {"0.mp4": 0.1, "1.mp4": 0.3, "2.mp4": 0.2, "3.mp4": 0.4}
            return [
                (torch.tensor([1 - scores[sample.video_path], scores[sample.video_path]]).log(), sample.label)
                for sample in samples
            ]

        samples = [
            FightSample(f"{index}.mp4", index % 2, "scfd" if index < 2 else "airtlab", f"g{index}", split)
            for split in ("val", "test") for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            manifest = root / "manifest.csv"
            checkpoint.write_bytes(b"mock checkpoint")
            manifest.write_bytes(b"mock manifest")
            with (
                patch("campus_safety_ai.apps.evaluate_fight.read_manifest", return_value=samples),
                patch("campus_safety_ai.apps.evaluate_fight.FightTsn", return_value=Model()),
                patch("campus_safety_ai.apps.evaluate_fight.FightVideoDataset", side_effect=dataset),
                patch("torch.load", return_value={
                    "frame_count": 8, "image_size": 224, "model_name": "test-model", "model_state": {}
                }),
            ):
                val_path = root / "val.json"
                val = evaluate(root, manifest, checkpoint, "val", val_path, device="cpu", select_threshold=True)
                self.assertAlmostEqual(val["threshold"], 0.25)
                self.assertEqual(val["metrics"]["balanced_accuracy"], 1)
                self.assertEqual(val["confusion_matrix"]["true_negative"], 2)
                self.assertEqual(len(val["predictions"]), 4)
                self.assertFalse(val["partial"])
                self.assertEqual(val["checkpoint_sha256"], _sha256(checkpoint))
                self.assertEqual(val["manifest_sha256"], _sha256(manifest))
                for name in ("airtlab", "scfd"):
                    self.assertEqual(val["by_dataset"][name]["sample_count"], 2)
                    self.assertEqual(val["by_dataset"][name]["metrics"]["f1"], 1)
                self.assertIn("decode", val["timing"]["scope"])
                self.assertGreater(val["timing"]["samples_per_second"], 0)
                self.assertEqual(json.loads(val_path.read_text(encoding="utf-8")), val)

                test = evaluate(
                    root, manifest, checkpoint, "test", root / "test.json", device="cpu", threshold_report=val_path
                )
                self.assertEqual(test["threshold"], val["threshold"])
                self.assertEqual(test["threshold_source"]["kind"], "validation_report")
                self.assertIsNone(test["threshold_selection"])
                self.assertEqual(test["metrics"], val["metrics"])

                partial = evaluate(
                    root, manifest, checkpoint, "val", root / "partial.json", device="cpu", max_samples=2
                )
                self.assertTrue(partial["partial"])
                self.assertEqual(partial["available_sample_count"], 4)
                self.assertEqual(partial["threshold"], 0.5)
                self.assertEqual(len(partial["misclassified"]), 1)
                with self.assertRaisesRegex(ValueError, "complete"):
                    evaluate(
                        root, manifest, checkpoint, "val", root / "invalid.json", device="cpu",
                        max_samples=2, select_threshold=True,
                    )


if __name__ == "__main__":
    unittest.main()
