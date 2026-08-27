import json
import tempfile
import unittest
from pathlib import Path

from campus_safety_ai.apps.fight_replay import run


class FightReplayTests(unittest.TestCase):
    def test_golden_replay_matches_expected_lifecycle(self) -> None:
        source = Path(__file__).parent / "golden/fight/observations.jsonl"
        expected_path = Path(__file__).parent / "golden/fight/expected-events.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            self.assertEqual(run(source, output), 2)
            actual = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        projection = [
            {
                "eventType": record["eventType"],
                "phase": record["phase"],
                "observedAt": record["observedAt"],
            }
            for record in actual
        ]
        self.assertEqual(projection, expected)
        self.assertEqual(actual[0]["eventId"], actual[1]["eventId"])


if __name__ == "__main__":
    unittest.main()

