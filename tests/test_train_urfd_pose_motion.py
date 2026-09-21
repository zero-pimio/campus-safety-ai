import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("train_urfd_pose_motion", Path(__file__).resolve().parents[1]
                                             / "scripts/train_urfd_pose_motion.py")
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def make_fixture(root):
    samples = []
    for prefix, count, label, train, val in (("fall", 30, 1, 18, 6), ("adl", 40, 0, 24, 8)):
        names = [f"{prefix}-{i:02d}" for i in range(1, count + 1)]
        random.Random(20260921).shuffle(names)
        for i, name in enumerate(names):
            samples.append({"sequence_id": name, "group_id": f"urfd:{name}", "label": label,
                            "split": "train" if i < train else "val" if i < train + val else "test",
                            "frame_timestamps_ms": [float(j * 100) for j in range(16)]})
    samples.sort(key=lambda row: row["sequence_id"])
    manifest = root / "manifest.json"
    trainer.write_json(manifest, {"dataset": "urfd-cam0", "seed": 20260921, "frame_count": 16,
                                  "class_names": ["non_fall", "fall"], "samples": samples})
    cache = root / "cache"
    cache.mkdir()
    identity = {"schema_version": "1.0", "manifest_sha256": trainer.sha256(manifest),
                "pose_checkpoint_sha256": "a" * 64, "settings": {"test_fixture": True},
                "source_sha256": trainer.sha256(trainer.ROOT / "scripts/extract_urfd_pose.py"), "device": "cpu"}
    trainer.write_json(cache / "cache-identity.json", identity)
    return manifest, cache, identity, samples


def add_cache(cache, identity, sample, valid=True):
    points = np.zeros((16, 17, 3), dtype=np.float64)
    points[:, :, 2] = .9 if valid else 0
    base = np.zeros((17, 2))
    base[5], base[6], base[11], base[12] = [40, 30], [60, 30], [45, 60], [55, 60]
    boxes = np.tile([20, 10, 80, 100], (16, 1)).astype(float)
    for i in range(16):
        shift = np.array([i * .2, i * (2 if sample["label"] else .01)])
        points[i, :, :2] = base + shift
        boxes[i] += np.tile(shift, 2)
    trainer.write_json(cache / f"{sample['sequence_id']}.json", {
        "schema_version": "1.0", "manifest_sha256": identity["manifest_sha256"],
        "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"],
        "cache_identity_sha256": trainer.sha256(cache / "cache-identity.json"),
        "sequence_id": sample["sequence_id"], "label": sample["label"], "split": sample["split"],
        "keypoints": points.tolist(), "boxes": boxes.tolist(),
        "timestamps_ms": sample["frame_timestamps_ms"], "tracking": {"fixture": True}})


class PoseThresholdTests(unittest.TestCase):
    def test_static_gate_is_strict_and_does_not_change_real_denominators(self):
        selection = trainer.select_threshold([0, 0, 1, 1], [.1, .3, .6, .9], .5)
        self.assertTrue(selection["eligible"])
        self.assertGreater(selection["threshold"], .5)
        self.assertEqual(selection["real_validation_sample_count"], 4)
        self.assertEqual(selection["real_metrics"]["balanced_accuracy"], 1)
        self.assertEqual(selection["real_metrics"]["f1"], 1)

    def test_gate_cannot_be_rescued_by_clipping_saturated_static_score(self):
        result = trainer.select_threshold([0, 1], [.1, .9], 1)
        self.assertFalse(result["eligible"])
        self.assertNotIn("threshold", result)

    def test_threshold_input_requires_both_valid_real_classes(self):
        for labels, scores, zero in (([0], [.1], .1), ([0, 1], [.1], .1), ([0, 1], [.1, .9], float("nan"))):
            with self.subTest(labels=labels, zero=zero), self.assertRaises(ValueError):
                trainer.select_threshold(labels, scores, zero)


class PoseTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_manifest_rejects_group_leakage_and_changed_prespecified_split(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _, samples = make_fixture(Path(directory))
            loaded, _ = trainer.read_manifest(manifest)
            self.assertEqual(len(loaded["samples"]), 70)
            for field, value in (("group_id", "wrong"), ("split", "test")):
                broken = json.loads(manifest.read_text())
                sample = next(x for x in broken["samples"] if x["split"] == "train")
                sample[field] = value
                path = Path(directory) / f"bad-{field}.json"
                trainer.write_json(path, broken)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    trainer.read_manifest(path)
            self.assertEqual(len(samples), 70)

    def test_invalid_pose_is_retained_unknown_never_static_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, cache, identity, samples = make_fixture(Path(directory))
            sample = next(s for s in samples if s["split"] == "val")
            add_cache(cache, identity, sample, valid=False)
            rows, controls, _ = trainer.read_cohort([sample], cache, trainer.sha256(manifest), identity)
            self.assertFalse(rows[0]["valid"])
            self.assertIsNone(rows[0]["features"])
            self.assertTrue(all(not control["valid"] for control in controls))
            model = trainer.PoseMotionClassifier(32)
            with patch.object(model, "forward", side_effect=AssertionError("unknown must not be inferred")):
                report = trainer.evaluate(model, rows, .5)
            self.assertEqual(report["coverage"], 0)
            self.assertIsNone(report["metrics"])
            self.assertEqual(report["unknown_sample_count"], 1)
            self.assertEqual(report["predictions"][0]["status"], "unknown")
            self.assertIsNone(report["predictions"][0]["prediction"])

    def test_nonempty_output_is_protected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                trainer.train_phase(output / "absent", output / "cache", output)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_train_freezes_before_test_and_preserves_coverage_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cache, identity, samples = make_fixture(root)
            development = [s for s in samples if s["split"] != "test"]
            for sample in development:
                add_cache(cache, identity, sample)
            output = root / "output"
            # Real tiny heads, descriptor, optimizer, serialization and ONNX run;
            # only the epoch budget is shortened for this synthetic fixture.
            with patch.object(trainer, "MAX_EPOCHS", 8), patch.object(trainer, "PATIENCE", 3):
                frozen = trainer.train_phase(manifest, cache, output)
            self.assertFalse((output / "test-attempt.json").exists())
            self.assertFalse((output / "test.json").exists())
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(len(protocol["development_cache_sha256"]), 56)
            self.assertTrue(all(name not in protocol["development_cache_sha256"]
                                for name in protocol["split_ids"]["test"]))
            checkpoint = torch.load(output / "best.pt", weights_only=True)
            descriptors = json.loads((output / "development-descriptors.json").read_text())["records"]
            train = [r for r in descriptors if r["split"] == "train"]
            features, _ = trainer.valid_tensors(train)
            torch.testing.assert_close(checkpoint["model_state"]["feature_mean"], features.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(checkpoint["model_state"]["feature_scale"],
                                       features.std(0, unbiased=False).clamp_min(trainer.SCALE_FLOOR), rtol=0, atol=0)
            for hidden in (0, 16):
                candidate = json.loads((output / f"candidate-hidden-{hidden}.json").read_text())
                self.assertGreater(candidate["parameter_delta"]["changed_parameter_count"], 0)
                self.assertTrue(all(abs(row["total_loss"] - row["real_weighted_ce"]
                                            - .25 * row["synthetic_zero_ce"]) < 1e-6 for row in candidate["history"]))
            gate = json.loads((output / "static-gate.json").read_text())
            self.assertTrue(gate["eligible"])
            self.assertEqual(gate["unique_valid_descriptor_count"], 1)
            self.assertEqual(gate["real_sample_count"], 0)
            self.assertEqual(gate["false_positive_count"], 0)
            # No test data existed during fitting. Publish the exact fixed test
            # cohort now, with one unknown to ensure abstention survives reporting.
            test_samples = [s for s in samples if s["split"] == "test"]
            for index, sample in enumerate(test_samples):
                add_cache(cache, identity, sample, valid=index != 0)
            result = trainer.test_phase(manifest, cache, output)
            self.assertTrue(result["reused_test_split"])
            self.assertEqual((result["available_sample_count"], result["valid_sample_count"],
                              result["unknown_sample_count"]), (14, 13, 1))
            self.assertEqual(len(result["predictions"]), 14)
            self.assertTrue(result["partial"])
            self.assertEqual(result["threshold"], frozen["threshold"])
            saved = (output / "test.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already"):
                trainer.test_phase(manifest, cache, output)
            self.assertEqual((output / "test.json").read_bytes(), saved)
            # Frozen artifacts are checked even when only reloading for audit.
            (output / "static-gate.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                trainer.verify_frozen(manifest, cache, output)


if __name__ == "__main__":
    unittest.main()
