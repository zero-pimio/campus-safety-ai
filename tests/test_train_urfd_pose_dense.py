import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("train_urfd_pose_dense", Path(__file__).resolve().parents[1]
                                             / "scripts/train_urfd_pose_dense.py")
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


def fixture(cache):
    samples, annotations, payloads = [], {"fall_sequences": []}, {}
    count = 76
    for name in dense.DENSE_IDS:
        label = int(name.startswith("fall"))
        sample = {"sequence_id": name, "group_id": f"urfd:{name}", "split": "train",
                  "label": label, "sync_sha256": "a" * 64}
        points = np.zeros((count, 17, 3))
        points[:, :, 2] = .9
        base = np.zeros((17, 2))
        base[5], base[6], base[11], base[12] = [40, 30], [60, 30], [45, 60], [55, 60]
        boxes = np.tile([20, 10, 80, 100], (count, 1)).astype(float)
        for i in range(count):
            shift = np.array([i * .2, i * .1])
            points[i, :, :2] = base + shift
            boxes[i] += np.tile(shift, 2)
        raw = {**sample, "keypoints": points.tolist(), "boxes": boxes.tolist(),
               "timestamps_ms": [float(i * 100) for i in range(count)],
               "frame_numbers": list(range(1, count + 1)),
               "tracking": [{"track_id": 1, "association_valid": True} for _ in range(count)]}
        (cache / f"{name}.json").write_text(json.dumps(raw))
        samples.append(sample)
        payloads[name] = raw
        if label:
            annotations["fall_sequences"].append({
                **sample, "posture_intervals": [{"posture_label": 0, "first_timestamp_ms": 4000},
                                                {"posture_label": 1, "first_timestamp_ms": 5000}]})
    return samples, annotations, payloads


def parent_rows(records, name):
    return {row["prefix_length"]: row for row in records if row["parent_sequence_id"] == name}


class DensePrefixTests(unittest.TestCase):
    def test_only_train_samples_and_matching_raw_identity_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            samples, annotations, payloads = fixture(cache)
            for split in ("val", "test"):
                changed = copy.deepcopy(samples)
                changed[0]["split"] = split
                with self.subTest(sample_split=split), self.assertRaisesRegex(ValueError, "validation or test"):
                    dense.dense_prefixes(changed, cache, annotations)
            for field, value in (("split", "val"), ("group_id", "urfd:adl-99"),
                                  ("sequence_id", "adl-99"), ("label", 1)):
                raw = copy.deepcopy(payloads["adl-01"])
                raw[field] = value
                (cache / "adl-01.json").write_text(json.dumps(raw))
                with self.subTest(raw_field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                    dense.dense_prefixes(samples, cache, annotations)

    def test_bad_times_and_frame_order_fail_before_descriptor_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            samples, annotations, payloads = fixture(cache)
            for case in ("negative", "duplicate", "backward", "nan", "infinite", "short",
                         "duplicate_frame", "backward_frame"):
                raw = copy.deepcopy(payloads["adl-01"])
                if case == "negative":
                    raw["timestamps_ms"][0] = -1
                elif case in ("duplicate", "backward"):
                    raw["timestamps_ms"][5] = raw["timestamps_ms"][4] - int(case == "backward")
                elif case in ("nan", "infinite"):
                    raw["timestamps_ms"][5] = float("nan" if case == "nan" else "inf")
                elif case == "short":
                    raw["timestamps_ms"] = raw["timestamps_ms"][:15]
                    raw["frame_numbers"] = raw["frame_numbers"][:15]
                else:
                    raw["frame_numbers"][5] = raw["frame_numbers"][4] - int(case == "backward_frame")
                (cache / "adl-01.json").write_text(json.dumps(raw))
                with self.subTest(case=case), self.assertRaisesRegex(ValueError, "invalid dense observation"):
                    dense.dense_prefixes(samples, cache, annotations)

    def test_label_margins_use_latest_selected_input_and_future_frames_cannot_change_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            samples, annotations, payloads = fixture(cache)
            # At length 64, equal-bin sampling ends at index 62 (6200ms),
            # while the observed endpoint is index 63 (6300ms).
            for zero, one, expected in ((6400, 9000, None), (6500, 9000, 0),
                                         (3000, 6000, 1), (3000, 6100, None)):
                changed = copy.deepcopy(annotations)
                target = next(row for row in changed["fall_sequences"] if row["sequence_id"] == "fall-01")
                target["posture_intervals"] = [{"posture_label": 0, "first_timestamp_ms": zero},
                                               {"posture_label": 1, "first_timestamp_ms": one}]
                records, _ = dense.dense_prefixes(samples, cache, changed)
                fall = parent_rows(records, "fall-01")
                with self.subTest(first_zero=zero, first_one=one):
                    if expected is None:
                        self.assertNotIn(64, fall)
                    else:
                        self.assertEqual(fall[64]["label"], expected)
                        self.assertEqual(fall[64]["latest_input_timestamp_ms"], 6200)
                        self.assertEqual(fall[64]["latest_input_frame_number"], 63)
                    for row in records:
                        self.assertLess(max(row["selected_indices"]), row["prefix_length"])
                        self.assertEqual(row["split"], "train")
            original, _ = dense.dense_prefixes(samples, cache, annotations)
            before = parent_rows(original, "adl-01")[64]
            raw = copy.deepcopy(payloads["adl-01"])
            for i in range(64, len(raw["keypoints"])):
                raw["keypoints"][i] = [[10000, -10000, 0] for _ in range(17)]
                raw["boxes"][i] = [1000, 2000, 3000, 4000]
                raw["tracking"][i] = {"track_id": 99, "association_valid": False}
            (cache / "adl-01.json").write_text(json.dumps(raw))
            changed, _ = dense.dense_prefixes(samples, cache, annotations)
            after = parent_rows(changed, "adl-01")[64]
            self.assertEqual(after, before)

    def test_unselected_association_breaks_mask_observations_and_can_make_prefix_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            samples, annotations, payloads = fixture(cache)
            baseline, _ = dense.dense_prefixes(samples, cache, annotations)
            clean = parent_rows(baseline, "adl-01")[64]
            self.assertTrue(clean["valid"])
            self.assertNotIn(4, clean["selected_indices"])
            self.assertEqual(clean["intervening_association_masked_positions"], [])
            for case in ("association_missing", "track_switch", "all_gaps_missing"):
                raw = copy.deepcopy(payloads["adl-01"])
                if case == "track_switch":
                    raw["tracking"][4]["track_id"] = 2
                else:
                    for index in ([4] if case == "association_missing" else range(4, 61, 4)):
                        raw["tracking"][index]["association_valid"] = False
                (cache / "adl-01.json").write_text(json.dumps(raw))
                records, _ = dense.dense_prefixes(samples, cache, annotations)
                row = parent_rows(records, "adl-01")[64]
                with self.subTest(case=case):
                    self.assertIn(1, row["intervening_association_masked_positions"])
                    self.assertFalse(row["quality"]["valid_frame_mask"][1])
                    self.assertFalse(row["quality"]["valid_pair_mask"][0])
                    self.assertFalse(row["quality"]["valid_pair_mask"][1])
                    if case == "all_gaps_missing":
                        self.assertFalse(row["valid"])
                        self.assertIsNone(row["features"])
                        self.assertEqual(row["quality"]["valid_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
