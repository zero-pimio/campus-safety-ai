import unittest

from campus_safety_ai.adapters.video_sources import redact_video_source
from campus_safety_ai.apps.fight_video import _source_stem


class VideoSourceSafetyTests(unittest.TestCase):
    def test_rtsp_credentials_are_not_rendered(self) -> None:
        source = "rtsp://camera-user:camera-password@10.0.0.8:554/live/main?token=secret"

        self.assertEqual(redact_video_source(source), "rtsp://10.0.0.8:554/live/main")
        self.assertEqual(_source_stem(source, "fallback"), "main")


if __name__ == "__main__":
    unittest.main()
