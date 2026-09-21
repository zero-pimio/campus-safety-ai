import copy
import hashlib
import importlib.util
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


class UrfdBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script_path = Path(__file__).resolve().parents[1] / "scripts/train_urfd_baseline.py"
        spec = importlib.util.spec_from_file_location("urfd_baseline_under_test", script_path)
        cls.script = importlib.util.module_from_spec(spec)
        try:
            for dependency in ("numpy", "torch", "torchvision", "PIL.Image"):
                importlib.import_module(dependency)
        except ImportError:
            raise unittest.SkipTest("optional URFD torch/torchvision/Pillow dependencies are not installed") from None
        spec.loader.exec_module(cls.script)
        cls.torch = cls.script.torch
        cls.original_threads = cls.torch.get_num_threads()
        cls.torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.manifest = {"dataset": "urfd-cam0", "seed": cls.script.SEED, "frame_count": 16,
                        "class_names": ["non_fall", "fall"], "samples": []}
        for prefix, count, train_count, val_count, label in (("fall", 30, 18, 6, 1), ("adl", 40, 24, 8, 0)):
            ordered = [f"{prefix}-{index:02d}" for index in range(1, count + 1)]
            random.Random(cls.script.SEED).shuffle(ordered)
            for index, sequence in enumerate(ordered):
                split = "train" if index < train_count else "val" if index < train_count + val_count else "test"
                frames, members = [], []
                for frame_index in range(16):
                    relative = f"frames/{sequence}/{frame_index:02d}.bin"
                    path = cls.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"unique-frame-bytes:{sequence}:{frame_index}".encode()
                    path.write_bytes(payload)
                    frames.append(relative)
                    members.append({"sha256": hashlib.sha256(payload).hexdigest()})
                cls.manifest["samples"].append({
                    "sequence_id": sequence, "group_id": f"urfd:{sequence}", "label": label, "split": split,
                    "frames": frames, "members": members, "frame_numbers": list(range(1, 17)),
                    "frame_timestamps_ms": [index * 40 for index in range(16)],
                })

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.data = copy.deepcopy(self.manifest)

    def split_sample(self, split, label=0):
        return next(sample for sample in self.data["samples"] if sample["split"] == split and sample["label"] == label)

    def test_complete_frozen_manifest_accepts_all_70_sequences(self):
        samples = self.script.validate_manifest(self.data, self.root)
        self.assertEqual(len(samples), 70)
        self.assertEqual(Counter(sample["split"] for sample in samples), {"train": 42, "val": 14, "test": 14})

    def test_manifest_rejects_cross_split_group_reuse(self):
        train, validation = self.split_sample("train"), self.split_sample("val")
        validation["group_id"] = train["group_id"]
        with self.assertRaisesRegex(ValueError, "group"):
            self.script.validate_manifest(self.data, self.root)

    def test_manifest_rejects_cross_split_frame_path_alias(self):
        train, validation = self.split_sample("train"), self.split_sample("val")
        validation["frames"][0] = "./" + train["frames"][0]
        validation["members"][0] = copy.deepcopy(train["members"][0])
        with self.assertRaisesRegex(ValueError, "duplicate frame path"):
            self.script.validate_manifest(self.data, self.root)

    def test_manifest_rejects_identical_frame_bytes_at_different_paths_across_splits(self):
        train, validation = self.split_sample("train"), self.split_sample("val")
        target = self.root / validation["frames"][0]
        original = target.read_bytes()
        try:
            target.write_bytes((self.root / train["frames"][0]).read_bytes())
            validation["members"][0]["sha256"] = self.script.sha256(target)
            with self.assertRaisesRegex(ValueError, "identical frame bytes cross data splits"):
                self.script.validate_manifest(self.data, self.root)
        finally:
            target.write_bytes(original)

    def test_manifest_rejects_file_or_expected_sha_tampering(self):
        sample = self.split_sample("train")
        target = self.root / sample["frames"][0]
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.script.validate_manifest(self.data, self.root)
        finally:
            target.write_bytes(original)
        sample["members"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.script.validate_manifest(self.data, self.root)

    def test_manifest_rejects_membership_swap_even_when_split_counts_are_unchanged(self):
        train, validation = self.split_sample("train"), self.split_sample("val")
        original_counts = Counter((sample["split"], sample["label"]) for sample in self.data["samples"])
        train["split"], validation["split"] = validation["split"], train["split"]
        self.assertEqual(Counter((sample["split"], sample["label"]) for sample in self.data["samples"]), original_counts)
        with self.assertRaisesRegex(ValueError, "split membership differs"):
            self.script.validate_manifest(self.data, self.root)

    def test_normalization_uses_only_supplied_training_features_and_population_std(self):
        torch = self.torch

        class SummaryModel:
            def summarize_features(self, features):
                self.received = features.clone()
                return features

            def set_feature_normalization(self, mean, scale):
                self.mean, self.scale = mean.clone(), scale.clone()

        model = SummaryModel()
        train = torch.tensor([[1., 4., .01], [3., 4., .02], [5., 4., .03]])
        all_features = torch.cat((train, torch.full((2, 3), 10000.)))
        self.script.fit_normalization(model, all_features[:3])
        expected_mean = train.mean(0)
        expected_scale = train.std(0, unbiased=False).clamp_min(.05)
        torch.testing.assert_close(model.received, train, rtol=0, atol=0)
        torch.testing.assert_close(model.mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.scale, expected_scale, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.mean, all_features.mean(0)))
        all_features[3:] = -1000000
        self.script.fit_normalization(model, all_features[:3])
        torch.testing.assert_close(model.mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.scale, expected_scale, rtol=0, atol=0)
        self.script.fit_normalization(model, train[:1])
        torch.testing.assert_close(model.scale, torch.full((3,), .05), rtol=0, atol=0)

    def test_load_frames_preserves_rgb_and_letterboxes_sixteen_real_pngs(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(16):
                name = f"{index:02d}.png"
                size, color = ((8, 4), (255, 0, 0)) if index % 2 == 0 else ((4, 8), (0, 255, 0))
                self.script.Image.new("RGB", size, color).save(root / name)
                paths.append(name)
            tensor = self.script.load_frames({"frames": paths}, root)
        self.assertEqual(tuple(tensor.shape), (1, 16, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        mean = torch.tensor(self.script.PREPROCESS["mean"])
        std = torch.tensor(self.script.PREPROCESS["std"])
        padding = (torch.tensor(self.script.PREPROCESS["padding_rgb"]) / 255 - mean) / std
        red = (torch.tensor([1., 0., 0.]) - mean) / std
        green = (torch.tensor([0., 1., 0.]) - mean) / std
        torch.testing.assert_close(tensor[0, 0, :, 55, 112], padding)
        torch.testing.assert_close(tensor[0, 0, :, 56, 112], red)
        torch.testing.assert_close(tensor[0, 1, :, 112, 55], padding)
        torch.testing.assert_close(tensor[0, 1, :, 112, 56], green)

    def test_candidates_reset_to_identical_initial_head(self):
        torch = self.torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 2)
                self.seen_heads = []

            def forward_features(self, features):
                self.seen_heads.append(copy.deepcopy(self.head.state_dict()))
                return self.head(features)

        torch.manual_seed(19)
        model = TinyModel()
        initial = copy.deepcopy(model.head.state_dict())
        features = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        samples = [{"label": label} for label in (0, 1, 0, 1)]
        fixed_validation = {"metrics": {"loss": 1., "balanced_accuracy": .5, "f1": .5}}
        with tempfile.TemporaryDirectory() as directory, patch.object(self.script, "evaluate", return_value=fixed_validation):
            for weight_decay in (.01, .1):
                with torch.no_grad():
                    model.head.weight.fill_(999)
                    model.head.bias.fill_(999)
                first_forward = len(model.seen_heads)
                self.script.train_candidate(model, initial, features, samples, features, samples,
                                            weight_decay, Path(directory))
                for key, value in initial.items():
                    torch.testing.assert_close(model.seen_heads[first_forward][key], value, rtol=0, atol=0)
