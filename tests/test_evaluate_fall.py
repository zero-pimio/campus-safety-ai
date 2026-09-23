import json
import tempfile
import unittest
from pathlib import Path

from campus_safety_ai.apps.evaluate_fall import evaluate


class EvaluateFallTests(unittest.TestCase):
    def inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        truth, events, observations, output = [root / name for name in ("truth.json", "events.jsonl", "obs.jsonl", "report.json")]
        truth.write_text(json.dumps({
            "sessions": [{"cameraId": "cam", "sourceEpoch": 0, "startedAt": "2026-09-23T00:00:00Z", "endedAt": "2026-09-23T01:00:00Z"}],
            "events": [],
        }), encoding="utf-8")
        events.write_text("", encoding="utf-8")
        observations.write_text("", encoding="utf-8")
        return truth, events, observations, output

    def test_writes_real_report_and_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory))
            report = evaluate(*inputs)
            loaded = json.loads(inputs[-1].read_text(encoding="utf-8"))
            self.assertEqual(loaded, report)
            self.assertEqual(loaded["metrics"]["annotated_hours"], 1)
            self.assertEqual(loaded["coverage"]["unobserved_seconds"], 3600)
            with self.assertRaises(FileExistsError):
                evaluate(*inputs)

    def test_malformed_jsonl_has_line_number_and_does_not_write_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory))
            inputs[2].write_text("\n[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obs.jsonl:2"):
                evaluate(*inputs)
            self.assertFalse(inputs[-1].exists())

    def test_json_nonfinite_constant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory))
            inputs[2].write_text('{"score": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                evaluate(*inputs)
            self.assertFalse(inputs[-1].exists())


if __name__ == "__main__":
    unittest.main()
