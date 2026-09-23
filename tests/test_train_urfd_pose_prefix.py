import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch  # noqa: F401 - the loaded training script requires torch
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch"}:
        raise
    raise unittest.SkipTest(f"optional {error.name} dependency is not installed") from error

SPEC = importlib.util.spec_from_file_location("train_urfd_pose_prefix", Path(__file__).resolve().parents[1]
                                             / "scripts/train_urfd_pose_prefix.py")
prefix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefix)


def fixture(cache, *, label=1, eligible=8, invalid_index=None, jump_after_gap=False):
    name = "fall-01" if label else "adl-01"
    sample = {"sequence_id": name, "label": label, "split": "train", "group_id": f"urfd:{name}",
              "frame_numbers": list(range(10, 170, 10)), "frame_timestamps_ms": [i * 100.0 for i in range(16)],
              "sync_sha256": "b" * 64}
    points = np.zeros((16, 17, 3))
    points[:, :, 2] = .9
    base = np.zeros((17, 2))
    base[5], base[6], base[11], base[12] = [40, 30], [60, 30], [45, 60], [55, 60]
    boxes = np.tile([20, 10, 80, 100], (16, 1)).astype(float)
    for index in range(16):
        shift = np.array([index + (1000 if jump_after_gap and index >= 3 else 0), index * .1])
        points[index, :, :2] = base + shift
        boxes[index] += np.tile(shift, 2)
    if invalid_index is not None:
        points[invalid_index, :, 2] = 0
    (cache / f"{name}.json").write_text(json.dumps({
        "sequence_id": name, "label": label, "split": "train",
        "keypoints": points.tolist(), "boxes": boxes.tolist(), "timestamps_ms": sample["frame_timestamps_ms"],
    }))
    annotation = {"sequence_id": name, "group_id": sample["group_id"], "split": "train",
                  "sync_sha256": sample["sync_sha256"],
                  "sampled_frames": [{"frame_number": number, "sample_index": index,
                                      "timestamp_ms": sample["frame_timestamps_ms"][index],
                                      "before_first_zero": index < eligible,
                                      "posture_label": -1 if index < eligible else 0}
                                     for index, number in enumerate(sample["frame_numbers"])]}
    return sample, {"manifest_sha256": "a" * 64, "fall_sequences": [annotation] if label else []}


class NegativePrefixTests(unittest.TestCase):
    def test_validation_and_test_groups_are_rejected_before_cache_read(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            sample, annotations = fixture(cache)
            (cache / "fall-01.json").unlink()
            for split in ("val", "test"):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "only from train"):
                    prefix.negative_prefixes([{**sample, "split": split}], cache, annotations)

    def test_annotation_frame_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            sample, annotations = fixture(cache)
            for variant in ("different_frame", "different_order", "missing_frame"):
                broken = copy.deepcopy(annotations)
                rows = broken["fall_sequences"][0]["sampled_frames"]
                if variant == "different_frame":
                    rows[0]["frame_number"] += 1
                elif variant == "different_order":
                    rows[0], rows[1] = rows[1], rows[0]
                else:
                    rows.pop()
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "frame correspondence"):
                    prefix.negative_prefixes([sample], cache, broken)
            for field, value, message in (("group_id", "urfd:fall-02", "group or split"),
                                          ("split", "val", "group or split"),
                                          ("sync_sha256", "c" * 64, "synchronization"),
                                          ("timestamp_ms", 1.0, "timestamps")):
                broken = copy.deepcopy(annotations)
                annotation = broken["fall_sequences"][0]
                if field == "timestamp_ms":
                    annotation["sampled_frames"][0][field] = value
                else:
                    annotation[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                    prefix.negative_prefixes([sample], cache, broken)

    def test_fall_margin_drops_one_actual_sample_and_never_restarts_after_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for eligible in (0, 3, 4, 5, 8, 16):
                sample, annotations = fixture(cache, eligible=eligible)
                records, audit = prefix.negative_prefixes([sample], cache, annotations)
                maximum = min(15, eligible - 1)
                with self.subTest(eligible=eligible):
                    self.assertEqual(audit[0]["maximum_prefix_length"], maximum)
                    self.assertEqual([r["prefix_length"] for r in records], list(range(4, maximum + 1)))
                    for row in records:
                        self.assertEqual(row["label"], 0)
                        self.assertEqual(row["original_sequence_label"], 1)
                        self.assertEqual(row["group_id"], sample["group_id"])
                        self.assertIn("proxy", row["label_basis"])
                        self.assertEqual(row["end_frame_number"], sample["frame_numbers"][row["prefix_length"] - 1])
                    if records:
                        self.assertLess(records[-1]["end_frame_number"], sample["frame_numbers"][eligible - 1])
            sample, annotations = fixture(cache, eligible=8)
            # Later -1 annotations must not restart a prefix after an observed transition.
            annotations["fall_sequences"][0]["sampled_frames"][4]["posture_label"] = 0
            records, _ = prefix.negative_prefixes([sample], cache, annotations)
            self.assertEqual(records, [])
            sample, annotations = fixture(cache, label=0)
            records, _ = prefix.negative_prefixes([sample], cache, annotations)
            self.assertEqual([r["prefix_length"] for r in records], list(range(4, 16)))
            self.assertTrue(all(row["original_sequence_label"] == 0 for row in records))

    def test_invalid_observation_is_not_removed_or_reconnected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            sample, annotations = fixture(cache, label=0, invalid_index=2)
            clean, _ = prefix.negative_prefixes([sample], cache, annotations)
            short = next(row for row in clean if row["prefix_length"] == 5)
            self.assertEqual(short["quality"]["valid_frame_count"], 4)
            self.assertEqual(short["quality"]["valid_pair_count"], 2)
            self.assertEqual(short["quality"]["valid_pair_mask"], [True, False, False, True])
            self.assertFalse(short["valid"])
            self.assertIsNone(short["features"])
            longer = next(row for row in clean if row["prefix_length"] == 6)
            self.assertTrue(longer["valid"])
            self.assertEqual(longer["quality"]["valid_pair_count"], 3)
            # An arbitrary jump across the invalid frame must contribute no velocity.
            sample, annotations = fixture(cache, label=0, invalid_index=2, jump_after_gap=True)
            jumped, _ = prefix.negative_prefixes([sample], cache, annotations)
            jumped_longer = next(row for row in jumped if row["prefix_length"] == 6)
            np.testing.assert_allclose(jumped_longer["features"], longer["features"], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
