import importlib.util
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campus_safety_ai.training.manifest import FightSample
from campus_safety_ai.training.video_dataset import FightVideoDataset, _read_sampled_frames, _sample_frames


class VideoDatasetContractTests(unittest.TestCase):
    def test_dataset_and_module_do_not_require_optional_imports_until_access(self):
        code = """
import builtins
original = builtins.__import__
def optional_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cv2', 'numpy'}:
        raise ImportError('optional dependency unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = optional_import
from pathlib import Path
from campus_safety_ai.training.video_dataset import FightVideoDataset
assert len(FightVideoDataset(Path('.'), [])) == 0
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_dataset_round_trips_through_pickle(self):
        sample = FightSample("video.avi", 1, "synthetic", "clip-1", "train")
        dataset = FightVideoDataset(Path("/tmp"), [sample], frame_count=4, image_size=32, training=True)

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertIs(type(restored), FightVideoDataset)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.samples, (sample,))
        self.assertEqual(restored.project_root, Path("/tmp"))
        self.assertEqual((restored.frame_count, restored.image_size, restored.training), (4, 32, True))

    def test_invalid_dimensions_fail_before_opening_video(self):
        for frame_count, image_size in ((0, 32), (-1, 32), (4, 0), (4, -2), (2.5, 32), (4, True)):
            with self.subTest(frame_count=frame_count, image_size=image_size):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    FightVideoDataset(Path("."), [], frame_count, image_size)
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    _sample_frames(Path("missing.avi"), frame_count, image_size, False)

    def test_invalid_frame_count_metadata_releases_capture(self):
        for count in (float("nan"), float("inf"), -float("inf"), 0, -1, 0.5):
            with self.subTest(count=count):
                capture = SimpleNamespace(isOpened=lambda: True, get=lambda _, count=count: count)
                released = []
                capture.release = lambda released=released: released.append(True)
                cv2 = SimpleNamespace(VideoCapture=lambda _, capture=capture: capture, CAP_PROP_FRAME_COUNT=1)
                with patch.dict(sys.modules, {"cv2": cv2, "numpy": SimpleNamespace()}):
                    with self.assertRaisesRegex(ValueError, "positive finite frame count"):
                        _sample_frames(Path("invalid.avi"), 4, 32, False)
                self.assertEqual(released, [True])

    def test_capture_is_released_when_metadata_access_raises(self):
        released = []

        def fail(_):
            raise OSError("metadata unavailable")

        capture = SimpleNamespace(isOpened=lambda: True, get=fail,
                                  release=lambda: released.append(True))
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FRAME_COUNT=1)
        with patch.dict(sys.modules, {"cv2": cv2, "numpy": SimpleNamespace()}):
            with self.assertRaisesRegex(OSError, "metadata unavailable"):
                _sample_frames(Path("invalid.avi"), 4, 32, False)
        self.assertEqual(released, [True])

    def test_unopened_capture_is_released(self):
        released = []
        capture = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture)
        with patch.dict(sys.modules, {"cv2": cv2, "numpy": SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, "cannot open video"):
                _sample_frames(Path("invalid.avi"), 4, 32, False)
        self.assertEqual(released, [True])

    def test_sequential_decoder_reuses_duplicates_and_seeks_for_large_gaps(self):
        class Capture:
            position = 0
            grabs = 0

            def __init__(self):
                self.seeks = []
                self.reads = []

            def set(self, prop, position):
                self.position = position
                self.seeks.append(position)
                return True

            def grab(self):
                self.grabs += 1
                self.position += 1
                return True

            def read(self):
                self.reads.append(self.position)
                frame = bytearray([self.position % 256])
                self.position += 1
                return True, frame

        capture = Capture()
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(CAP_PROP_POS_FRAMES=1)}):
            frames = list(_read_sampled_frames(capture, [0, 128, 257, 257, 500], Path("video.avi")))
        self.assertEqual(capture.reads, [0, 128, 257, 500])
        self.assertEqual(capture.seeks, [257, 500])
        self.assertEqual(capture.grabs, 127)
        self.assertEqual(frames[2], frames[3])
        self.assertIsNot(frames[2], frames[3])


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("cv2", "numpy", "torch")),
                     "training dependencies are not installed")
class VideoDatasetRuntimeTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(random.setstate, random.getstate())
        self.root = Path(self.directory.name)
        self.video = self.root / "synthetic.avi"
        x, y = np.meshgrid(np.arange(80), np.arange(64))
        self.frame = np.stack((x * 3, y * 3, (x + y) % 128), axis=2).astype(np.uint8)
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 12, (80, 64))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(12):
                writer.write(self.frame)
        finally:
            writer.release()
        self.samples = [FightSample("synthetic.avi", 1, "synthetic", "clip-1", "train")]

    def test_temporal_augmentation_is_shared_and_seed_is_reproducible(self):
        import numpy as np
        import torch

        dataset = FightVideoDataset(self.root, self.samples, frame_count=4, image_size=24, training=True)
        random.seed(41)
        first, label = dataset[0]
        random.seed(41)
        second, _ = pickle.loads(pickle.dumps(dataset))[0]

        self.assertEqual(tuple(first.shape), (12, 24, 24))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(label, 1)
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        frames = first.numpy().reshape(4, 3, 24, 24)
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame, frames[0])

    def test_validation_keeps_segment_centers_and_center_crop(self):
        import cv2
        import numpy as np

        positions, released = [], []
        next_index = 0

        def grab():
            nonlocal next_index
            next_index += 1
            return True

        def read():
            nonlocal next_index
            positions.append(next_index)
            next_index += 1
            return True, self.frame.copy()

        capture = SimpleNamespace(
            isOpened=lambda: True, get=lambda _: 12,
            grab=grab, read=read,
            release=lambda: released.append(True),
        )
        with patch("cv2.VideoCapture", return_value=capture):
            actual = _sample_frames(self.video, 4, 24, False)

        self.assertEqual(positions, [1, 4, 7, 10])
        self.assertEqual(released, [True])
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (70, 56), interpolation=cv2.INTER_LINEAR)
        crop = resized[16:40, 23:47].transpose(2, 0, 1).astype(np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        expected = (crop - mean) / std
        for frame in actual.reshape(4, 3, 24, 24):
            np.testing.assert_array_equal(frame, expected)

    def test_decode_failure_releases_capture(self):
        released = []
        capture = SimpleNamespace(
            isOpened=lambda: True, get=lambda _: 12,
            grab=lambda: True, read=lambda: (False, None),
            release=lambda: released.append(True),
        )
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "cannot decode frame"):
                _sample_frames(self.video, 4, 24, False)
        self.assertEqual(released, [True])

    def test_eof_while_skipping_frames_releases_capture(self):
        released = []
        capture = SimpleNamespace(
            isOpened=lambda: True, get=lambda _: 12, grab=lambda: False,
            release=lambda: released.append(True),
        )
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "cannot decode frame 0"):
                _sample_frames(self.video, 4, 24, False)
        self.assertEqual(released, [True])

    def test_large_gap_seek_failure_releases_capture(self):
        released = []
        capture = SimpleNamespace(
            isOpened=lambda: True, get=lambda _: 2000, set=lambda *_: False,
            release=lambda: released.append(True),
        )
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "cannot seek to frame 249"):
                _sample_frames(self.video, 4, 24, False)
        self.assertEqual(released, [True])

    def test_real_videos_match_legacy_seek_with_training_and_repeated_indices(self):
        import cv2
        import numpy as np

        def legacy_read(capture, indices, path):
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                available, frame = capture.read()
                if not available:
                    raise ValueError(f"cannot decode frame {index}: {path}")
                yield frame

        for frame_total in (2, 12):
            video = self.root / f"temporal-{frame_total}.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 12, (80, 64))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(frame_total):
                    frame = self.frame.copy()
                    frame[:, :, 2] += index * 5
                    writer.write(frame)
            finally:
                writer.release()
            for training in (False, True):
                with self.subTest(frame_total=frame_total, training=training):
                    random.seed(57)
                    actual = _sample_frames(video, 8, 24, training)
                    random.seed(57)
                    with patch("campus_safety_ai.training.video_dataset._read_sampled_frames", legacy_read):
                        expected = _sample_frames(video, 8, 24, training)
                    np.testing.assert_array_equal(actual, expected)

    def test_spawn_worker_reads_real_synthetic_video(self):
        import torch
        from torch.utils.data import DataLoader

        dataset = FightVideoDataset(self.root, self.samples, frame_count=4, image_size=24)
        loader = DataLoader(dataset, batch_size=1, num_workers=1,
                            multiprocessing_context="spawn", timeout=30)

        batches = list(loader)

        self.assertEqual(len(batches), 1)
        frames, labels = batches[0]
        self.assertEqual(tuple(frames.shape), (1, 12, 24, 24))
        self.assertEqual(labels.tolist(), [1])
        self.assertTrue(torch.isfinite(frames).all())
        model = torch.nn.Sequential(
            torch.nn.Conv2d(12, 2, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(model(frames), labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
