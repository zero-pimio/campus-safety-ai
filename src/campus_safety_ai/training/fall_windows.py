"""Auditable training rows replayed through the production causal window sampler."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from campus_safety_ai.adapters.runtimes.pose_fall import sha256
from campus_safety_ai.adapters.timed_file_source import validate_timestamps
from campus_safety_ai.contracts import iso_time
from campus_safety_ai.core.event_metrics import evaluate_fall_events
from campus_safety_ai.core.fall_analysis import FallEventAnalysis, FallObservation, FallPolicy
from campus_safety_ai.core.fall_pipeline import FallPipeline, PoseFrame, WindowPolicy
from campus_safety_ai.training.pose_motion import motion_descriptor

ORIGIN = datetime(2000, 1, 1, tzinfo=UTC)  # relative replay anchor, never a claimed capture date


def window_label(start: float, end: float, intervals: list[dict], minimum_motion: float = .2) -> int | None:
    if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
            for t in (start, end, minimum_motion)) or end <= start or minimum_motion <= 0):
        raise ValueError("window and motion threshold must be finite and positive")
    positive = sum(max(0, min(end, row["end_s"]) - max(start, row["start_s"]))
                   for row in intervals if row["training_label"] == 1)
    if positive >= minimum_motion - 1e-6:
        return 1
    negative = sum(max(0, min(end, row["end_s"]) - max(start, row["start_s"]))
                   for row in intervals if row["training_label"] == 0)
    return 0 if negative >= end - start - 1e-6 else None


def validate_manifest(manifest: dict) -> None:
    owners = {key: {} for key in ("subject_id", "source_group", "sha256")}
    ids = set()
    for sample in manifest["samples"]:
        if sample["sample_id"] in ids:
            raise ValueError("duplicate sample id")
        ids.add(sample["sample_id"])
        if sample["split"] not in {"train", "val", "test", "external_diagnostic"}:
            raise ValueError("unassigned development sample")
        for key, mapping in owners.items():
            value = sample[key]
            if not value or (value in mapping and mapping[value] != sample["split"]):
                raise ValueError(f"missing identity or cross-split leakage: {key}")
            mapping[value] = sample["split"]
        duration = sample["timing"]["duration_s"]
        times = sample["timing"]["frame_timestamps_s"]
        validate_timestamps(times)
        if not math.isfinite(duration) or duration <= 0 or times[-1] > duration:
            raise ValueError("invalid annotation duration")
        previous = 0.0
        for row in sample["intervals"]:
            if not previous - 1e-6 <= row["start_s"] < row["end_s"] <= sample["timing"]["duration_s"] + 1e-6:
                raise ValueError("invalid or overlapping annotation intervals")
            if row["training_label"] not in (None, 0, 1):
                raise ValueError("invalid training label")
            previous = row["end_s"]


class DescriptorHead:
    model_version = "descriptor-only"

    def __init__(self):
        self.features = None

    def predict(self, frames):
        origin = frames[0].observed_at
        descriptor = motion_descriptor(np.asarray([f.keypoints for f in frames]),
                                       np.asarray([f.box for f in frames]),
                                       [(f.observed_at - origin).total_seconds() * 1000 for f in frames],
                                       min_conf=.25)
        self.features = descriptor["features"].tolist() if descriptor["valid"] else None
        return (0.0 if descriptor["valid"] else None), descriptor["quality"]


def collect_windows(sample: dict, cache_dir: Path, policy: WindowPolicy) -> list[dict]:
    path = cache_dir / (sample["sample_id"] + ".json")
    prepared = json.loads((cache_dir / "prepared.json").read_text())
    if (prepared["identity_sha256"] != sha256(cache_dir / "identity.json")
            or prepared["records"].get(sample["sample_id"]) != sha256(path)):
        raise ValueError("pose cache body differs from prepared fingerprint")
    cache = json.loads(path.read_text())
    if (cache["sample_id"] != sample["sample_id"] or cache["source_sha256"] != sample["sha256"]
            or cache["identity_sha256"] != sha256(cache_dir / "identity.json")
            or cache["timestamps_s"] != sample["timing"]["frame_timestamps_s"]
            or len(cache["frames"]) != len(cache["timestamps_s"])):
        raise ValueError("pose cache does not match source and timing")
    head = DescriptorHead()
    flow = FallPipeline(head, FallEventAnalysis(FallPolicy(edge_id="training-extraction")), policy)
    rows = []
    for expected, (raw, seconds) in enumerate(zip(cache["frames"], cache["timestamps_s"], strict=True), 1):
        if raw["sequence"] != expected:
            raise ValueError("pose cache skipped or duplicated a frame")
        frame = PoseFrame(sample["sample_id"], 1, expected, ORIGIN + timedelta(seconds=seconds),
                          raw["keypoints"], raw["box"], raw["track_key"], raw["valid"], raw["reason"])
        head.features = None
        batch = flow.advance(frame)
        if batch.observation is None:
            continue
        observation = batch.observation
        start = (observation.window_started_at - ORIGIN).total_seconds()
        rows.append({"sample_id": sample["sample_id"], "split": sample["split"], "seconds": seconds,
                     "label": window_label(start, seconds, sample["intervals"]) if seconds > start else None,
                     "features": head.features, "observation": observation.to_dict()})
    return rows


def event_report(samples: list[dict], rows: list[dict], scores: list[float | None], policy: FallPolicy) -> dict:
    if len(rows) != len(scores):
        raise ValueError("scores must match observation rows")
    sessions, truths, events, observations = [], [], [], []
    by_id = {}
    sample_ids = {sample["sample_id"] for sample in samples}
    for row, score in zip(rows, scores, strict=True):
        if (row["sample_id"] not in sample_ids or row["observation"]["cameraId"] != row["sample_id"]
                or row["observation"]["sourceEpoch"] != 1):
            raise ValueError("evaluation observation belongs to a different sample or source epoch")
        by_id.setdefault(row["sample_id"], []).append((row, score))
    for sample in samples:
        camera = sample["sample_id"]  # unique replay stream, NOT an asserted physical camera
        sessions.append({"cameraId": camera, "sourceEpoch": 1, "startedAt": iso_time(ORIGIN),
                         "endedAt": iso_time(ORIGIN + timedelta(seconds=sample["timing"]["duration_s"]))})
        for index, event in enumerate(sample["events"]):
            truths.append({"cameraId": camera, "sourceEpoch": 1, "eventId": f"{camera}:{index}",
                           "startedAt": iso_time(ORIGIN + timedelta(seconds=event["start_s"])),
                           "endedAt": iso_time(ORIGIN + timedelta(seconds=event["end_s"]))})
        analyzer = FallEventAnalysis(policy)
        for row, score in by_id.get(camera, []):
            raw = {**row["observation"], "score": score, "modelVersion": "candidate-replay"}
            observation = FallObservation.from_dict(raw)
            observations.append(raw)
            for event in analyzer.advance(observation):
                events.append({**event.to_dict(), "sourceEpoch": 1})
        for event in analyzer.finalize(camera, 1, ORIGIN + timedelta(seconds=sample["timing"]["frame_timestamps_s"][-1])):
            events.append({**event.to_dict(), "sourceEpoch": 1})
    # Strict matching plus separately reported fixed 0.5 s boundary/confirmation allowance.
    return {"policy": asdict(policy),
            "strict": evaluate_fall_events({"sessions": sessions, "events": truths}, events, observations),
            "tolerance_0_5s": evaluate_fall_events({"sessions": sessions, "events": truths}, events, observations,
                                                tolerance_seconds=.5),
            "events": events, "observations": observations, "truth": {"sessions": sessions, "events": truths}}
