import math
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campus_safety_ai.adapters.frame_sources import OpenCvFileFrames


class FakeCapture:
    def __init__(self, count=6, *, declared_count=None, opened=True, metadata=None, read_error=None):
        self.frames = [SimpleNamespace(shape=(16, 24, 3), marker=index) for index in range(count)]
        self.metadata = {0: 4.0, 1: 24, 2: 16, 3: count if declared_count is None else declared_count}
        self.metadata.update(metadata or {})
        self.opened = opened
        self.read_error = read_error
        self.position = 0
        self.release_calls = 0

    def isOpened(self):
        return self.opened

    def get(self, key):
        value = self.metadata[key]
        if isinstance(value, Exception):
            raise value
        return value

    def read(self):
        if self.position == len(self.frames):
            if self.read_error is not None:
                raise self.read_error
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.release_calls += 1


class FrameSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "source.avi"
        self.path.touch()
        self.origin = datetime(2026, 9, 21, tzinfo=UTC)

    def source(self, capture, *, stride=1):
        cv2 = SimpleNamespace(
            VideoCapture=lambda path: capture, CAP_PROP_FPS=0, CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FRAME_COUNT=3,
        )
        with patch.dict("sys.modules", {"cv2": cv2}):
            return OpenCvFileFrames(self.path, "gate", 2, self.origin, frame_stride=stride)

    def test_stride_preserves_original_frame_numbers_and_media_time(self):
        capture = FakeCapture()
        source = self.source(capture, stride=2)
        frames = list(source)
        self.assertEqual([frame.sequence for frame in frames], [1, 3, 5])
        self.assertEqual([frame.image.marker for frame in frames], [0, 2, 4])
        self.assertEqual([frame.captured_at for frame in frames], [
            self.origin, self.origin + timedelta(seconds=.5), self.origin + timedelta(seconds=1),
        ])
        self.assertTrue(all((frame.camera_id, frame.source_epoch, frame.width, frame.height) ==
                            ("gate", 2, 24, 16) for frame in frames))
        self.assertTrue(all(frame.received_at.utcoffset() is not None for frame in frames))
        self.assertEqual(source.decoded_frames, 6)
        self.assertEqual(capture.release_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "once"):
            list(source)
        source.close()
        self.assertEqual(capture.release_calls, 1)

    def test_truncated_decode_is_error_and_releases_capture(self):
        capture = FakeCapture(count=1, declared_count=3)
        source = self.source(capture)
        frames = iter(source)
        self.assertEqual(next(frames).sequence, 1)
        with self.assertRaisesRegex(ValueError, "1 of 3"):
            next(frames)
        self.assertEqual(capture.release_calls, 1)

    def test_decoder_exception_is_preserved_and_releases_capture(self):
        failure = OSError("bad packet")
        capture = FakeCapture(count=1, read_error=failure)
        frames = iter(self.source(capture))
        next(frames)
        with self.assertRaises(OSError) as caught:
            next(frames)
        self.assertIs(caught.exception, failure)
        self.assertEqual(capture.release_calls, 1)

    def test_context_manager_releases_on_early_consumer_failure(self):
        capture = FakeCapture()
        source = self.source(capture)
        with self.assertRaisesRegex(RuntimeError, "consumer failed"), source:
            frames = iter(source)
            next(frames)
            raise RuntimeError("consumer failed")
        self.assertEqual(capture.release_calls, 1)
        frames.close()
        self.assertEqual(capture.release_calls, 1)

    def test_dimension_change_releases_capture(self):
        capture = FakeCapture(count=2)
        capture.frames[1] = SimpleNamespace(shape=(32, 24, 3))
        frames = iter(self.source(capture))
        next(frames)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            next(frames)
        self.assertEqual(capture.release_calls, 1)

    def test_unopenable_source_releases_capture(self):
        capture = FakeCapture(opened=False)
        with self.assertRaisesRegex(ValueError, "open"):
            self.source(capture)
        self.assertEqual(capture.release_calls, 1)

    def test_invalid_or_unreadable_metadata_releases_capture(self):
        for metadata in ({0: math.nan}, {1: math.nan}, {2: math.inf}, {1: OSError("metadata read failed")}):
            with self.subTest(metadata=metadata):
                capture = FakeCapture(metadata=metadata)
                with self.assertRaises((ValueError, OverflowError, OSError)):
                    self.source(capture)
                self.assertEqual(capture.release_calls, 1)

    def test_real_short_avi_stride_and_timestamps(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are optional video dependencies")
        self.path.unlink()
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        if not writer.isOpened():
            writer.release()
            self.skipTest("MJPG writer unavailable")
        try:
            for index in range(6):
                writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
        finally:
            writer.release()
        with OpenCvFileFrames(self.path, "real-gate", 1, self.origin, frame_stride=2) as source:
            frames = list(source)
        self.assertEqual([frame.sequence for frame in frames], [1, 3, 5])
        self.assertEqual([frame.captured_at for frame in frames], [
            self.origin, self.origin + timedelta(seconds=.2), self.origin + timedelta(seconds=.4),
        ])
        self.assertEqual(source.decoded_frames, 6)
        for frame, expected in zip(frames, (0, 60, 120), strict=True):
            self.assertEqual(frame.image.shape, (24, 32, 3))
            self.assertAlmostEqual(float(frame.image.mean()), expected, delta=3)


if __name__ == "__main__":
    unittest.main()
