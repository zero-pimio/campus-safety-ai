import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from campus_safety_ai.adapters.evidence import NullEvidenceSink
from campus_safety_ai.apps.fight_video import run
from campus_safety_ai.core.event_delivery import InMemoryDestination
from campus_safety_ai.core.fight_analysis import FightEventAnalysis, FightPolicy
from campus_safety_ai.core.fight_inference import FightPrediction
from campus_safety_ai.core.fight_pipeline import FightVideoPipeline, VideoWindow


class Classifier:
    model_version = "stream-test"
    frame_len = 1

    def predict(self, frames):
        return FightPrediction((0.0, 1.0), 0.99)


def analysis():
    return FightEventAnalysis(FightPolicy(edge_id="edge", confirm_seconds=0))


class StreamingTests(unittest.TestCase):
    def test_first_event_is_available_before_source_finishes(self):
        class Source:
            fps = 10
            sample_frequency = 1

            def __iter__(self):
                yield VideoWindow(("frame",), 0, 5)
                raise RuntimeError("source still live")

        pipeline = FightVideoPipeline(Classifier(), analysis(), NullEvidenceSink())
        batches = pipeline.stream(Source(), "cam", 1, datetime.now(UTC))
        self.assertEqual([r.phase for r in next(batches).events], ["START"])
        with self.assertRaisesRegex(RuntimeError, "source still live"):
            next(batches)

    def test_cli_delivery_precedes_next_read_and_eof_closes_event(self):
        destination = InMemoryDestination()
        test = self

        class Source:
            fps = 10
            sample_frequency = 1
            closed = False

            def __init__(self, *args):
                pass

            def __iter__(self):
                yield VideoWindow(("frame",), 0, 5)
                test.assertEqual(destination.records[0]["phase"], "START")
                yield VideoWindow(("frame",), 6, 11)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                Source.closed = True

        with tempfile.TemporaryDirectory() as directory, patch(
            "campus_safety_ai.apps.fight_video.OpenCvVideoSource", Source
        ), patch("builtins.print"):
            result = run(
                "test.mp4", Path("unused"), Path(directory), "cam", datetime.now(UTC),
                classifier=Classifier(), analysis=analysis(), frame_len=1,
                destination=destination, collect_results=False,
            )
            self.assertEqual(result, ([], []))
            self.assertEqual([r["phase"] for r in destination.records], ["START", "END"])
            self.assertEqual(len((Path(directory) / "test-observations.jsonl").read_text().splitlines()), 2)
            self.assertTrue(Source.closed)

    def test_delivery_failure_closes_source_and_preserves_pending_event(self):
        import sqlite3

        class Source:
            fps = 10
            sample_frequency = 1
            closed = False

            def __init__(self, *args):
                pass

            def __iter__(self):
                yield VideoWindow(("frame",), 0, 5)
                raise AssertionError("must stop consuming after delivery failure")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                Source.closed = True

        class FailingDestination:
            def publish(self, record):
                raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory, patch(
            "campus_safety_ai.apps.fight_video.OpenCvVideoSource", Source
        ):
            with self.assertRaisesRegex(OSError, "offline"):
                run(
                    "test.mp4", Path("unused"), Path(directory), "cam", datetime.now(UTC),
                    classifier=Classifier(), analysis=analysis(), frame_len=1,
                    destination=FailingDestination(), collect_results=False,
                )
            self.assertTrue(Source.closed)
            with sqlite3.connect(Path(directory) / "test-events.sqlite3") as database:
                self.assertEqual(database.execute(
                    "SELECT COUNT(*) FROM outbox WHERE delivered=0"
                ).fetchone()[0], 1)

    def test_background_delivery_allows_video_to_advance_during_network_wait(self):
        import threading

        entered, release = threading.Event(), threading.Event()
        received = []
        test = self

        class Destination:
            def publish(self, record):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("source failed to advance")
                received.append(record["phase"])

        class Source:
            fps = 10
            sample_frequency = 1

            def __init__(self, *args):
                pass

            def __iter__(self):
                yield VideoWindow(("frame",), 0, 5)
                # The consumer is blocked in publish, while video reaches here.
                test.assertTrue(entered.wait(1))
                test.assertEqual(received, [])
                release.set()
                yield VideoWindow(("frame",), 6, 11)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                release.set()

        with tempfile.TemporaryDirectory() as directory, patch(
            "campus_safety_ai.apps.fight_video.OpenCvVideoSource", Source
        ), patch("builtins.print"):
            run("test.mp4", Path("unused"), Path(directory), "cam", datetime.now(UTC),
                classifier=Classifier(), analysis=analysis(), frame_len=1,
                destination=Destination(), background_delivery=True, collect_results=False)
            self.assertEqual(received, ["START", "END"])
            self.assertEqual(len((Path(directory) / "test-observations.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
