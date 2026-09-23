import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fall_runtime_benchmark", ROOT / "scripts/benchmark_fall_runtime.py")
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class BenchmarkFallRuntimeTests(unittest.TestCase):
    def test_frame_timings_and_throughput_have_correct_units_and_zero_semantics(self):
        stats = benchmark.TimingStats()
        self.assertIsNone(stats.summary()["mean_seconds"])
        for value in (.01, .02, .06):
            stats.observe(value)
        measured = benchmark.summarize_throughput(processed_frames=3, wall_seconds=.5,
                                                  video_seconds=.1, stages={"pose": stats})
        self.assertEqual(measured["frames_per_wall_second"], 6)
        self.assertEqual(measured["video_seconds_per_wall_second"], .2)
        self.assertAlmostEqual(measured["stages"]["pose"]["mean_seconds"], .03)
        self.assertEqual(measured["stages"]["pose"]["minimum_seconds"], .01)
        self.assertEqual(measured["stages"]["pose"]["maximum_seconds"], .06)
        zero = benchmark.summarize_throughput(processed_frames=0, wall_seconds=0, video_seconds=0, stages={})
        self.assertIsNone(zero["frames_per_wall_second"])
        self.assertIsNone(zero["video_seconds_per_wall_second"])
        for value in (float("nan"), float("inf"), -.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stats.observe(value)

    def test_video_exposure_uses_variable_pts_and_explicit_final_duration(self):
        timing = benchmark.decoded_timing_from_probe({"frames": [
            {"best_effort_timestamp_time": "0.033333", "duration_time": "0.032"},
            {"best_effort_timestamp_time": "0.065333", "duration_time": "0.064"},
            {"best_effort_timestamp_time": "0.129333", "duration_time": "0.020"},
        ]})
        self.assertEqual(timing["timestamps_s"], [0, .032, .096])
        self.assertEqual(timing["frame_durations_s"], [.032, .064, .020])
        self.assertEqual(timing["duration_s"], .116)
        self.assertEqual(timing["original_first_pts_s"], .033333)
        self.assertAlmostEqual(sum(timing["frame_durations_s"]), timing["duration_s"])
        # A budget truncation after two frames accounts for only their known exposures.
        self.assertAlmostEqual(sum(timing["frame_durations_s"][:2]), .096)

    def test_missing_final_duration_and_nonmonotonic_pts_are_rejected(self):
        bad_payloads = [
            {"frames": [{"best_effort_timestamp_time": "0"}]},
            {"frames": [{"best_effort_timestamp_time": "0", "duration_time": "nan"}]},
            {"frames": [{"best_effort_timestamp_time": "0", "duration_time": ".03"},
                        {"best_effort_timestamp_time": "0", "duration_time": ".03"}]},
            {"frames": []},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                benchmark.decoded_timing_from_probe(payload)

    def test_peak_rss_converts_linux_kib_and_macos_bytes(self):
        self.assertEqual(benchmark.peak_rss_bytes(400, "Linux"), 409600)
        self.assertEqual(benchmark.peak_rss_bytes(400, "Darwin"), 400)
        with self.assertRaises(ValueError):
            benchmark.peak_rss_bytes(400, "unknown")

    def test_warmup_and_replay_loops_never_share_source_epochs(self):
        specs = list(benchmark.epoch_specs(9, 3, 2))
        self.assertEqual([item["source_epoch"] for item in specs], [9, 10, 11])
        self.assertEqual([item["kind"] for item in specs], ["warmup", "measured", "measured"])
        self.assertEqual(specs[0]["frame_limit"], 3)
        self.assertTrue(all(item["frame_limit"] is None for item in specs[1:]))
        self.assertEqual([item["source_epoch"] for item in benchmark.epoch_specs(9, 0, 2)], [9, 10])
        for args in ((0, 3, 2), (1, -1, 2), (1, 3, 0), (True, 3, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(benchmark.epoch_specs(*args))

    def test_missing_real_input_leaves_failed_report_without_hardware_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "benchmark"
            with self.assertRaisesRegex(ValueError, "required local input"):
                benchmark.run_benchmark(video=root / "missing.mp4", checkpoint=root / "missing.pt",
                                        detector=root / "pose.pt", event_config=root / "policy.toml",
                                        output_dir=output)
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["backend"], "pytorch-cpu")
            self.assertFalse(report["rknn_used"])
            self.assertEqual(report["epochs"], [])
            self.assertEqual(report["measurement"]["processed_frames"], 0)
            self.assertEqual(report["measurement"]["frames_per_wall_second"], None)
            self.assertIn("required local input", report["error"])
            self.assertGreater(report["peak_rss_bytes"], 0)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "preserve.txt"
            source.write_text("existing result")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                benchmark.run_benchmark(video=root / "missing.mp4", checkpoint=root / "missing.pt",
                                        detector=root / "pose.pt", event_config=root / "policy.toml",
                                        output_dir=root)
            self.assertEqual(source.read_text(), "existing result")
            self.assertFalse((root / "benchmark.json").exists())

    @unittest.skipUnless(os.getenv("FALL_BENCHMARK_REPORT"), "set FALL_BENCHMARK_REPORT to a real run report")
    def test_real_run_artifacts_account_for_frames_and_separate_loop_events(self):
        # Validate a separately executed real-model benchmark. No hardware or
        # model success is synthesized, mocked, or inferred from unit fixtures.
        path = Path(os.environ["FALL_BENCHMARK_REPORT"])
        report = json.loads(path.read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["backend"], "pytorch-cpu")
        self.assertFalse(report["rknn_used"])
        all_event_ids, samples = set(), []
        measured_epochs = [item for item in report["epochs"] if item["kind"] == "measured"]
        self.assertGreaterEqual(len(measured_epochs), 1)
        timing = json.loads(Path(report["inputs"]["decoded_timing"]["path"]).read_text())
        self.assertEqual(len({item["source_epoch"] for item in report["epochs"]}), len(report["epochs"]))
        for epoch in measured_epochs:
            directory = path.parent / f"measured-epoch-{epoch['source_epoch']}"
            with (directory / "frames.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), epoch["processed_frames"])
            self.assertEqual([int(row["sequence"]) for row in rows], list(range(1, len(rows) + 1)))
            self.assertEqual({int(row["source_epoch"]) for row in rows}, {epoch["source_epoch"]})
            samples.extend(rows)
            self.assertEqual([float(row["video_seconds"]) for row in rows],
                             timing["timestamps_s"][:len(rows)])
            self.assertEqual([float(row["frame_video_duration_seconds"]) for row in rows],
                             timing["frame_durations_s"][:len(rows)])
            self.assertAlmostEqual(epoch["processed_video_seconds"],
                                   sum(timing["frame_durations_s"][:len(rows)]))
            if epoch["stop_reason"] == "source_end":
                self.assertEqual(len(rows), timing["frame_count"])
                self.assertAlmostEqual(epoch["processed_video_seconds"], timing["duration_s"])
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            ids = {event["eventId"] for event in events}
            self.assertFalse(ids & all_event_ids)
            all_event_ids.update(ids)
            self.assertTrue(all(event["sourceEpoch"] == epoch["source_epoch"] for event in events))
            self.assertEqual(epoch["open_events_at_exit"], 0)
            self.assertEqual(epoch["pending_at_exit"], 0)
        measured = report["measurement"]
        self.assertEqual(len(samples), measured["processed_frames"])
        self.assertAlmostEqual(measured["video_seconds"],
                               sum(float(row["frame_video_duration_seconds"]) for row in samples))
        self.assertAlmostEqual(measured["wall_seconds"], sum(epoch["wall_seconds"] for epoch in measured_epochs))
        for name, stats in measured["stages"].items():
            timings = [float(row[f"{name}_seconds"]) for row in samples]
            self.assertEqual(stats["count"], len(timings))
            self.assertAlmostEqual(stats["total_seconds"], sum(timings))
            self.assertAlmostEqual(stats["mean_seconds"], sum(timings) / len(timings))
        for item in report["inputs"].values():
            self.assertEqual(benchmark._identity(Path(item["path"]))["sha256"], item["sha256"])
