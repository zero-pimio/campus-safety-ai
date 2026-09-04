import unittest
from datetime import datetime, timezone

from campus_safety_ai.core.fight_analysis import FightEventAnalysis, FightPolicy
from campus_safety_ai.core.fight_inference import FightPrediction
from campus_safety_ai.core.fight_pipeline import FightVideoPipeline, VideoWindow


class FakeSource:
    fps = 10.0
    sample_frequency = 1

    def __iter__(self):
        yield VideoWindow(("frame",), 0, 5)


class AlwaysFight:
    model_version = "fake-fight"
    frame_len = 1

    def predict(self, rgb_frames):
        return FightPrediction((0.0, 1.0), 0.9)


class RecordingEvidence:
    def __init__(self):
        self.sequences = []

    def capture(self, observation, rgb_frames, sampled_fps):
        self.sequences.append(observation.sequence)
        return {"snapshot": "snapshot.jpg", "clip": "clip.mp4"}


class FailingEvidence:
    def capture(self, observation, rgb_frames, sampled_fps):
        raise OSError("disk full")


class FightPipelineTests(unittest.TestCase):
    def test_pipeline_finalizes_open_event_and_attaches_evidence(self) -> None:
        evidence = RecordingEvidence()
        pipeline = FightVideoPipeline(
            AlwaysFight(),
            FightEventAnalysis(
                FightPolicy(
                    edge_id="edge-01",
                    start_score=0.75,
                    end_score=0.35,
                    confirm_seconds=0,
                    clear_seconds=1,
                    cooldown_seconds=0,
                )
            ),
            evidence,
        )

        result = pipeline.run(
            FakeSource(), "camera-01", 7, datetime(2026, 8, 25, tzinfo=timezone.utc)
        )

        self.assertEqual([record.phase for record in result.events], ["START", "END"])
        self.assertEqual(result.events[0].event_id, result.events[1].event_id)
        self.assertEqual(
            result.events[0].evidence_uris, ("snapshot.jpg", "clip.mp4")
        )
        self.assertEqual(result.events[1].evidence_uris, result.events[0].evidence_uris)
        self.assertEqual(result.observations[0]["evidence"]["snapshot"], "snapshot.jpg")
        self.assertEqual(evidence.sequences, [1])

    def test_evidence_failure_does_not_suppress_event(self) -> None:
        pipeline = FightVideoPipeline(
            AlwaysFight(),
            FightEventAnalysis(
                FightPolicy(
                    edge_id="edge-01",
                    confirm_seconds=0,
                    clear_seconds=1,
                    cooldown_seconds=0,
                )
            ),
            FailingEvidence(),
        )

        result = pipeline.run(
            FakeSource(), "camera-01", 7, datetime(2026, 8, 25, tzinfo=timezone.utc)
        )

        self.assertEqual([record.phase for record in result.events], ["START", "END"])
        self.assertEqual(result.observations[0]["evidenceError"], "OSError: disk full")


if __name__ == "__main__":
    unittest.main()
