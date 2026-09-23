#!/usr/bin/env python3
"""Finite CPU reference benchmark: real video, pose model, events, and local delivery.

This measures the existing PyTorch path, not RKNN/NPU acceleration or accuracy.
No network source or external destination is supported.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class TimingStats:
    count: int = 0
    total_seconds: float = 0.0
    minimum_seconds: float | None = None
    maximum_seconds: float | None = None

    def observe(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("timing sample must be finite and nonnegative")
        self.count += 1
        self.total_seconds += seconds
        self.minimum_seconds = seconds if self.minimum_seconds is None else min(self.minimum_seconds, seconds)
        self.maximum_seconds = seconds if self.maximum_seconds is None else max(self.maximum_seconds, seconds)

    def summary(self) -> dict:
        return {**asdict(self), "mean_seconds": self.total_seconds / self.count if self.count else None}


def summarize_throughput(*, processed_frames: int, wall_seconds: float,
                         video_seconds: float, stages: dict[str, TimingStats]) -> dict:
    if (type(processed_frames) is not int or processed_frames < 0 or not math.isfinite(wall_seconds)
            or wall_seconds < 0 or not math.isfinite(video_seconds) or video_seconds < 0):
        raise ValueError("invalid throughput inputs")
    return {"processed_frames": processed_frames, "wall_seconds": wall_seconds,
            "video_seconds": video_seconds,
            "frames_per_wall_second": processed_frames / wall_seconds if wall_seconds > 0 else None,
            "video_seconds_per_wall_second": video_seconds / wall_seconds if wall_seconds > 0 else None,
            "stages": {name: value.summary() for name, value in stages.items()}}


def peak_rss_bytes(raw_value: float | None = None, system: str | None = None) -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_value is None else raw_value
    system = platform.system() if system is None else system
    if system not in {"Linux", "Darwin"}:
        raise ValueError("RSS unit conversion is supported only on Linux and macOS")
    if not math.isfinite(raw) or raw < 0:
        raise ValueError("invalid peak RSS")
    return int(raw * (1024 if system == "Linux" else 1))


def epoch_specs(source_epoch: int, warmup_frames: int, loops: int):
    if any(type(value) is not int for value in (source_epoch, warmup_frames, loops)) or (
            source_epoch < 1 or warmup_frames < 0 or loops < 1):
        raise ValueError("source_epoch and loops must be positive; warmup_frames must be nonnegative")
    epoch = source_epoch
    if warmup_frames:
        yield {"kind": "warmup", "source_epoch": epoch, "frame_limit": warmup_frames}
        epoch += 1
    for loop in range(1, loops + 1):
        yield {"kind": "measured", "source_epoch": epoch, "frame_limit": None, "loop": loop}
        epoch += 1



def decoded_timing_from_probe(payload: dict) -> dict:
    """Use decoded PTS spans; require explicit final exposure instead of guessing FPS."""
    from campus_safety_ai.adapters.timed_file_source import validate_timestamps

    frames = payload["frames"]
    raw = [float(row["best_effort_timestamp_time"]) for row in frames]
    if not raw or any(not math.isfinite(value) for value in raw):
        raise ValueError("decoded video timestamps are empty or nonfinite")
    timestamps = [round(value - raw[0], 6) for value in raw]
    validate_timestamps(timestamps)
    last_duration = float(frames[-1].get("duration_time", frames[-1].get("pkt_duration_time", 0)))
    if not math.isfinite(last_duration) or last_duration <= 0:
        raise ValueError("last decoded frame must have an explicit positive duration")
    durations = [round(b - a, 6) for a, b in zip(timestamps, timestamps[1:], strict=False)]
    durations.append(last_duration)
    return {"timestamps_s": timestamps, "frame_durations_s": durations, "frame_count": len(raw),
            "original_first_pts_s": raw[0], "duration_s": round(timestamps[-1] + last_duration, 6),
            "time_source": "ffprobe decoded best_effort_timestamp_time minus first PTS",
            "duration_basis": "Each nonfinal processed frame covers [PTS, next PTS); final frame uses "
                              "explicit decoded duration_time/pkt_duration_time. No mean-FPS approximation."}


def probe_decoded_timing(path: Path) -> dict:
    response = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time,duration_time,pkt_duration_time", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=120)
    return decoded_timing_from_probe(json.loads(response.stdout))


def _identity(path: Path) -> dict:
    import hashlib

    if not path.is_file():
        raise ValueError(f"required local input does not exist: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def _host() -> dict:
    versions = {}
    for name in ("torch", "ultralytics", "numpy", "opencv-python", "opencv-python-headless"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    board = None
    if platform.system() == "Linux":
        board_path = Path("/proc/device-tree/model")
        if board_path.is_file():
            board = board_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
    return {"system": platform.system(), "release": platform.release(), "architecture": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": sys.version,
            "packages": versions, "device_tree_model": board}


def run_benchmark(*, video: Path, checkpoint: Path, detector: Path, event_config: Path,
                  output_dir: Path, duration_seconds: float = 20, loops: int = 1,
                  warmup_frames: int = 5, source_epoch: int = 1, evidence: bool = True,
                  torch_threads: int | None = None) -> dict:
    """Write a complete or failed report; synchronous backend calls may exceed the soft deadline."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    specs = epoch_specs(source_epoch, warmup_frames, loops)
    first_spec = next(specs)  # Validate before creating output.
    if torch_threads is not None and (type(torch_threads) is not int or torch_threads < 1):
        raise ValueError("torch_threads must be a positive integer")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    tick = time.perf_counter()
    deadline = tick + duration_seconds
    report = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "host": _host(),
              "backend": "pytorch-cpu", "device": "cpu", "rknn_used": False,
              "scope": "Local-file CPU reference; no hardware acceptance, model accuracy, live-camera, "
                       "external-platform, multi-person coverage, or long-run stability claim.",
              "settings": {"duration_seconds": duration_seconds, "loops": loops,
                           "warmup_frames": warmup_frames, "source_epoch": source_epoch,
                           "evidence": evidence, "torch_threads_requested": torch_threads},
              "duration_semantics": "Soft total wall budget includes startup and warmup; checked at frame "
                                    "boundaries. One synchronous decode/inference/write can overrun it.",
              "rss_semantics": "Process lifetime peak RSS includes imports, models, warmup, and measurement; "
                               "not isolated accelerator memory or only the measured phase.",
              "epochs": [], "inputs": {}, "measurement": None, "stop_reason": None,
              "startup_seconds": None, "peak_rss_bytes": peak_rss_bytes()}
    stages = {name: TimingStats() for name in ("decode", "pose", "window_and_event_policy",
                                              "evidence_and_local_delivery", "frame_end_to_end")}
    measured_frames, measured_wall, measured_video = 0, 0.0, 0.0

    def save_report():
        report["total_wall_seconds"] = time.perf_counter() - tick
        report["peak_rss_bytes"] = peak_rss_bytes()
        report["measurement"] = summarize_throughput(processed_frames=measured_frames,
                                                     wall_seconds=measured_wall,
                                                     video_seconds=measured_video, stages=stages)
        report["frames_per_total_wall_second"] = measured_frames / report["total_wall_seconds"]
        target = output_dir / "benchmark.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    save_report()
    try:
        import itertools

        for name, path in {"video": video, "checkpoint": checkpoint, "detector": detector,
                           "event_config": event_config, "frozen_selection": checkpoint.parent / "frozen.json",
                           "training_protocol": checkpoint.parent / "protocol.json",
                           "motion_descriptor_source": PROJECT_ROOT / "src/campus_safety_ai/training/pose_motion.py"
                           }.items():
            report["inputs"][name] = _identity(path)
        if video.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
            raise ValueError("benchmark requires a local video file")
        import torch
        from campus_safety_ai.adapters.timed_file_source import OpenCvTimedFileFrames
        from campus_safety_ai.adapters.runtimes.pose_fall import FrozenPoseFallHead, SinglePersonPose
        from campus_safety_ai.apps.fall_video import load_policy
        from campus_safety_ai.core.fall_analysis import FallEventAnalysis
        from campus_safety_ai.core.fall_pipeline import FallPipeline, WindowPolicy
        from campus_safety_ai.core.event_delivery import EventDelivery, JsonlDestination
        from campus_safety_ai.core.retention import register_managed_evidence

        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        timing = probe_decoded_timing(video)
        timing_path = output_dir / "decoded-timing.json"
        timing_path.write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n")
        report["inputs"]["decoded_timing"] = _identity(timing_path)
        report["video_time_source"] = timing["time_source"]
        report["video_duration_basis"] = timing["duration_basis"]
        report["input_video_duration_seconds"] = timing["duration_s"]
        head = FrozenPoseFallHead(checkpoint, "cpu")
        policy = load_policy(event_config)
        window = WindowPolicy(**head.extraction_identity.get("window_policy", {}))
        report["model_version"] = head.model_version
        report["window"] = asdict(window)
        report["event_policy"] = asdict(policy)
        report["startup_seconds"] = time.perf_counter() - tick
        for spec in itertools.chain((first_spec,), specs):
            if time.perf_counter() >= deadline:
                report["stop_reason"] = "duration_budget_reached"
                break
            measured = spec["kind"] == "measured"
            epoch = {**spec, "status": "running", "processed_frames": 0, "observations": 0,
                     "unknown_observations": 0, "starts": 0, "ends": 0, "open_events_at_exit": None,
                     "pending_at_exit": None, "stop_reason": None}
            report["epochs"].append(epoch)
            directory = output_dir / f"{spec['kind']}-epoch-{spec['source_epoch']}"
            directory.mkdir()
            epoch_tick = time.perf_counter()
            # Both association history and event/window state are fresh for every loop.
            flow = FallPipeline(head, FallEventAnalysis(policy), window)
            perception = SinglePersonPose(detector, head, "cpu")
            epoch["pose_initialization_seconds"] = time.perf_counter() - epoch_tick
            active_evidence, active_events = {}, set()
            last_frame = None
            source = OpenCvTimedFileFrames(video, "benchmark-camera", spec["source_epoch"], datetime.now(UTC),
                                           timestamps=timing["timestamps_s"])
            epoch["input_width"], epoch["input_height"] = None, None
            epoch["input_pts_frame_count"] = len(source.timestamps)
            epoch["processed_video_seconds"] = 0.0
            epoch["source_epoch"] = source.source_epoch
            with source, EventDelivery(directory / "outbox.sqlite3",
                                      JsonlDestination(directory / "delivered-events.jsonl")) as delivery, (
                    directory / "observations.jsonl").open("x") as observations_file, (
                    directory / "events.jsonl").open("x") as events_file, (
                    directory / "frames.csv").open("x", newline="") as samples_file:
                writer = csv.DictWriter(samples_file, fieldnames=["source_epoch", "sequence", "video_seconds",
                                                                "frame_video_duration_seconds",
                                                                *[f"{name}_seconds" for name in stages],
                                                                "process_peak_rss_bytes"])
                writer.writeheader()

                def consume(batch, last_frame, *, epoch=epoch, active_events=active_events,
                            active_evidence=active_evidence, directory=directory):
                    if batch.observation is not None:
                        item = {**batch.observation.to_dict(), "quality": batch.quality}
                        observations_file.write(json.dumps(item, allow_nan=False) + "\n")
                        observations_file.flush()
                        epoch["observations"] += 1
                        epoch["unknown_observations"] += item["score"] is None
                    records = []
                    for record in batch.events:
                        if record.phase == "START":
                            epoch["starts"] += 1
                            active_events.add(record.event_id)
                            if evidence and last_frame is not None:
                                import cv2

                                path = directory / "evidence" / f"{record.event_id}.jpg"
                                path.parent.mkdir(exist_ok=True)
                                if not cv2.imwrite(str(path), last_frame.image):
                                    raise OSError("evidence snapshot writer returned false")
                                register_managed_evidence(path.parent, path)
                                active_evidence[record.event_id] = (str(path),)
                        record = replace(record, evidence_uris=active_evidence.get(record.event_id, ()))
                        if record.phase == "END":
                            epoch["ends"] += 1
                            active_events.discard(record.event_id)
                            active_evidence.pop(record.event_id, None)
                        events_file.write(json.dumps({**record.to_dict(), "sourceEpoch": batch.source_epoch,
                                                      "reason": batch.event_reasons.get(record.idempotency_key)},
                                                     allow_nan=False) + "\n")
                        records.append(record)
                    events_file.flush()
                    delivery.submit(records)

                try:
                    frames = iter(source)
                    while time.perf_counter() < deadline:
                        frame_tick = time.perf_counter()
                        try:
                            frame = next(frames)
                        except StopIteration:
                            epoch["stop_reason"] = "source_end"
                            break
                        last_frame = frame
                        epoch["input_width"], epoch["input_height"] = frame.width, frame.height
                        decoded = time.perf_counter()
                        pose = perception.predict(frame)
                        predicted = time.perf_counter()
                        batch = flow.advance(pose)
                        analyzed = time.perf_counter()
                        consume(batch, last_frame)
                        delivered = time.perf_counter()
                        sample = {"decode": decoded - frame_tick, "pose": predicted - decoded,
                                  "window_and_event_policy": analyzed - predicted,
                                  "evidence_and_local_delivery": delivered - analyzed,
                                  "frame_end_to_end": delivered - frame_tick}
                        writer.writerow({"source_epoch": frame.source_epoch, "sequence": frame.sequence,
                                         "video_seconds": timing["timestamps_s"][frame.sequence - 1],
                                         "frame_video_duration_seconds": timing["frame_durations_s"][frame.sequence - 1],
                                         **{f"{name}_seconds": value for name, value in sample.items()},
                                         "process_peak_rss_bytes": peak_rss_bytes()})
                        epoch["processed_frames"] += 1
                        frame_video_seconds = timing["frame_durations_s"][frame.sequence - 1]
                        epoch["processed_video_seconds"] += frame_video_seconds
                        if measured:
                            measured_frames += 1
                            measured_video += frame_video_seconds
                            for name, value in sample.items():
                                stages[name].observe(value)
                        if spec["frame_limit"] is not None and epoch["processed_frames"] >= spec["frame_limit"]:
                            epoch["stop_reason"] = "warmup_frame_limit"
                            break
                    if epoch["stop_reason"] is None:
                        epoch["stop_reason"] = "duration_budget_reached"
                    consume(flow.finalize(epoch["stop_reason"]), last_frame)
                    epoch["pending_at_exit"] = delivery.pending_count()
                    epoch["open_events_at_exit"] = len(active_events)
                    if epoch["processed_frames"] == 0:
                        raise ValueError("source produced no benchmark frames")
                    epoch["status"] = "completed"
                except BaseException:
                    epoch["status"] = "failed"
                    try:
                        consume(flow.finalize("source_error"), last_frame)
                    except BaseException as error:
                        epoch["finalize_error"] = f"{type(error).__name__}: {error}"
                    epoch["open_events_at_exit"] = len(active_events)
                    epoch["pending_at_exit"] = delivery.pending_count()
                    raise
                finally:
                    epoch["wall_seconds"] = time.perf_counter() - epoch_tick
                    if measured:
                        measured_wall += epoch["wall_seconds"]
                    epoch["peak_rss_bytes"] = peak_rss_bytes()
            save_report()
            if epoch["stop_reason"] == "duration_budget_reached":
                report["stop_reason"] = "duration_budget_reached"
                break
        report["torch_threads_observed"] = torch.get_num_threads()
        report["torch_interop_threads_observed"] = torch.get_num_interop_threads()
        if not measured_frames:
            raise ValueError("budget ended before any measured frames; increase duration or reduce warmup")
        report["status"] = "completed"
        report["stop_reason"] = report["stop_reason"] or "requested_loops_completed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        for epoch in report["epochs"]:
            if epoch["status"] == "running":
                epoch["status"] = report["status"]
        raise
    finally:
        save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "runtime/training/fall-pose-dense-v1-20260921/best.pt")
    parser.add_argument("--detector", type=Path, default=PROJECT_ROOT / "models/yolo26n-pose.pt")
    parser.add_argument("--event-config", type=Path, default=PROJECT_ROOT / "configs/events/fall-v1.toml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=20)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--source-epoch", type=int, default=1)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--no-evidence", action="store_true")
    args = parser.parse_args()
    try:
        report = run_benchmark(video=args.video, checkpoint=args.checkpoint, detector=args.detector,
                               event_config=args.event_config, output_dir=args.output_dir,
                               duration_seconds=args.duration_seconds, loops=args.loops,
                               warmup_frames=args.warmup_frames, source_epoch=args.source_epoch,
                               evidence=not args.no_evidence, torch_threads=args.torch_threads)
    except Exception as error:
        parser.exit(1, f"benchmark failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "backend": report["backend"],
                      "measurement": report["measurement"], "peak_rss_bytes": report["peak_rss_bytes"]}))
    print(f"report -> {args.output_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
