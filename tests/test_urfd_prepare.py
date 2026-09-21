import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_urfd", Path(__file__).resolve().parents[1]
                                             / "scripts/prepare_urfd.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class FakeHttp:
    def __init__(self, data, sync):
        self.data, self.sync = data, sync
        self.ranges = []

    def head(self, url):
        return {"url": url, "content_length": len(self.data), "etag": '"fixture-v1"', "last_modified": None}

    def range(self, identity, start, end):
        self.ranges.append((start, end))
        return self.data[start:end + 1]

    def small(self, url):
        return self.sync, {"url": url, "content_length": len(self.sync), "sha256": prepare.sha256(self.sync),
                           "etag": None, "last_modified": None}


class UrfdPrepareTests(unittest.TestCase):
    def test_split_is_sequence_disjoint_reproducible_and_stratified(self):
        first = prepare.split_plan()
        self.assertEqual(first, prepare.split_plan())
        self.assertEqual(len({row["group_id"] for row in first}), 70)
        self.assertEqual(Counter((row["split"], row["label"]) for row in first), {
            ("train", 1): 18, ("val", 1): 6, ("test", 1): 6,
            ("train", 0): 24, ("val", 0): 8, ("test", 0): 8,
        })
        self.assertEqual(prepare.sample_indices(160), list(range(5, 160, 10)))

    def test_synchronization_rejects_duplicate_missing_and_invalid_times(self):
        for data in (b"1,0\n1,33\n", b"1,nan\n", b"1,33\n2,0\n", b""):
            with self.subTest(data=data), self.assertRaises(ValueError):
                prepare.synchronization(data)
        self.assertEqual(prepare.synchronization(b"1,0,9.9\n2,33,8.8\n"), {1: 0, 2: 33})

    def test_ignored_range_is_rejected_without_reading_body(self):
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, size):
                raise AssertionError("must not consume ignored range")
        client = prepare.HttpClient()
        with patch.object(client, "_open", return_value=Response()), self.assertRaisesRegex(ValueError, "200"):
            client.range({"url": "https://example.test/data.zip", "content_length": 1000,
                          "etag": "fixture"}, 0, 99)

    def test_safe_members_reject_traversal_and_duplicate_frame_number(self):
        for names in (("../escape.png",), ("fall-01-cam0-rgb/fall-01-cam0-rgb-001.png",
                                         "fall-01-cam0-rgb/fall-01-cam0-rgb-1.png")):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for name in names:
                    archive.writestr(name, b"x")
            with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive, self.assertRaises(ValueError):
                prepare.png_members(archive, "fall-01")

    def fixture(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy required for PNG verification")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(1, 33):
                image = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                ok, png = cv2.imencode(".png", image)
                self.assertTrue(ok)
                archive.writestr(f"fall-01-cam0-rgb/fall-01-cam0-rgb-{index:03d}.png", png.tobytes())
        sync = "".join(f"{index},{(index - 1) * 33},1.0\n" for index in range(1, 33)).encode()
        return FakeHttp(buffer.getvalue(), sync)

    def test_preparation_checks_png_crc_and_resumes_without_image_downloads(self):
        client = self.fixture()
        planned = next(row for row in prepare.split_plan() if row["sequence_id"] == "fall-01")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = prepare.prepare_sequence(root, planned, client)
            self.assertEqual(sample["frame_numbers"], list(range(2, 33, 2)))
            self.assertEqual(sample["frame_timestamps_ms"], [(index - 1) * 33 for index in range(2, 33, 2)])
            self.assertEqual(len(sample["frames"]), 16)
            self.assertEqual((sample["width"], sample["height"]), (64, 64))
            for path, member in zip(sample["frames"], sample["members"], strict=True):
                self.assertEqual(prepare.sha256((root / path).read_bytes()), member["sha256"])
            requests = len(client.ranges)
            self.assertEqual(prepare.prepare_sequence(root, planned, client), sample)
            self.assertEqual(len(client.ranges), requests)
            image = root / sample["frames"][0]
            image.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "CRC"):
                prepare.prepare_sequence(root, planned, client)
            self.assertEqual(image.read_bytes(), b"corrupt")

    def test_missing_selected_sync_frame_refuses_guessing(self):
        client = self.fixture()
        client.sync = b"1,0,1\n"
        planned = next(row for row in prepare.split_plan() if row["sequence_id"] == "fall-01")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "missing"):
            prepare.prepare_sequence(Path(temporary), planned, client)

    def test_missing_tail_sync_excludes_frames_and_uses_only_actual_timestamps(self):
        client = self.fixture()
        actual_times = {index: index * 37 + index % 3 for index in range(1, 31)}
        client.sync = "".join(f"{index},{timestamp}\n" for index, timestamp in actual_times.items()).encode()
        planned = next(row for row in prepare.split_plan() if row["sequence_id"] == "fall-01")
        with tempfile.TemporaryDirectory() as temporary:
            sample = prepare.prepare_sequence(Path(temporary), planned, client)
            expected = [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30]
            self.assertEqual(sample["frame_numbers"], expected)
            self.assertEqual(sample["member_count"], 32)
            self.assertEqual(sample["sampling_population_count"], 30)
            self.assertEqual(sample["excluded_unsynchronized_frame_numbers"], [31, 32])
            self.assertEqual(sample["frame_timestamps_ms"], [actual_times[index] for index in expected])
            self.assertEqual(sample["sync_sha256"], prepare.sha256(client.sync))
            self.assertEqual(len(sample["frames"]), 16)

    def test_interrupted_sequence_reuses_completed_pngs_on_resume(self):
        client = self.fixture()
        planned = next(row for row in prepare.split_plan() if row["sequence_id"] == "fall-01")
        original_range = client.range

        def interrupted(identity, start, end):
            if len(client.ranges) == 4:
                raise OSError("transient connection failure")
            return original_range(identity, start, end)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(client, "range", side_effect=interrupted), self.assertRaisesRegex(OSError, "transient"):
                prepare.prepare_sequence(root, planned, client)
            directory = root / prepare.DATA_PATH / "fall-01"
            cached = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.glob("*.png")}
            self.assertGreater(len(cached), 0)
            self.assertFalse((directory / "sample.json").exists())
            sample = prepare.prepare_sequence(root, planned, client)
            self.assertEqual(len(sample["frames"]), 16)
            for name, (data, modified) in cached.items():
                self.assertEqual((directory / name).read_bytes(), data)
                self.assertEqual((directory / name).stat().st_mtime_ns, modified)

    def test_unknown_file_and_changed_remote_identity_are_not_overwritten(self):
        client = self.fixture()
        planned = next(row for row in prepare.split_plan() if row["sequence_id"] == "fall-01")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = prepare.prepare_sequence(root, planned, client)
            directory = (root / sample["frames"][0]).parent
            unknown = directory / "my-notes.txt"
            unknown.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                prepare.prepare_sequence(root, planned, client)
            self.assertEqual(unknown.read_text(), "preserve")
            unknown.unlink()
            identity_path = directory / "archive.json"
            identity = json.loads(identity_path.read_text())
            identity["etag"] = "changed"
            with patch.object(client, "head", return_value=identity), self.assertRaisesRegex(ValueError, "overwrite"):
                prepare.prepare_sequence(root, planned, client)


if __name__ == "__main__":
    unittest.main()
