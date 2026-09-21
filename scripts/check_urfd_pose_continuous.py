"""Check frozen pose motion predictions on two predetermined causal video prefixes.

This is an out-of-distribution process check, not event recall or alarm latency:
these sequences have clip labels and no frame-level transition ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/private/urfd-continuous-v1"
POSE_CHECKPOINT = ROOT / "models/yolo26n-pose.pt"
EXPECTED_POSE_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
SEQUENCES = {"adl-10": (300, 0), "fall-06": (100, 1)}
NOTE = ("Causal-prefix out-of-distribution process check; no frame-level truth. "
        "First positive is an observed timestamp, not detection delay. "
        "No event recall or alarm latency is measured; no threshold fitting is performed.")


def extractor_module():
    path = Path(__file__).with_name("extract_urfd_pose.py")
    spec = importlib.util.spec_from_file_location("continuous_pose_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(extractor) -> dict[str, dict]:
    """Verify every original image before loading a detector or classifier."""
    samples = {}
    for sequence, (count, label) in SEQUENCES.items():
        path = DATA / sequence / "metadata.json"
        sample = json.loads(path.read_text())
        if any(sample.get(key) != value for key, value in {
            "sequence_id": sequence, "split": "val", "label": label, "frame_count": count,
        }.items()):
            raise ValueError(f"unexpected predetermined sequence metadata: {path}")
        fields = ("frames", "members", "frame_numbers", "frame_timestamps_ms")
        if any(len(sample[key]) != count for key in fields):
            raise ValueError(f"misaligned metadata: {path}")
        times = np.asarray(sample["frame_timestamps_ms"], dtype=np.float64)
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError(f"invalid official timestamps: {path}")
        paths = []
        for name, member, number, timestamp in zip(*(sample[k] for k in fields), strict=True):
            frame = (ROOT / name).resolve()
            if not frame.is_relative_to((DATA / sequence).resolve()) or frame.suffix.lower() != ".png":
                raise ValueError(f"unexpected original frame path: {frame}")
            if (member["path"] != name or member["frame_number"] != number
                    or member["timestamp_ms"] != timestamp):
                raise ValueError(f"member identity mismatch: {frame}")
            if frame.stat().st_size != member["size"] or extractor.sha256(frame) != member["sha256"]:
                raise ValueError(f"original frame SHA/size mismatch: {frame}")
            paths.append(str(frame))
        if len(set(paths)) != count or len(set(sample["frame_numbers"])) != count:
            raise ValueError(f"duplicate original frames: {path}")
        sample.update(metadata_sha256=extractor.sha256(path), absolute_frames=paths)
        samples[sequence] = sample
    return samples


def expected_record(sequence: str, sample: dict, identity_sha: str) -> dict:
    return {"schema_version": "1.0", "sequence_id": sequence, "split": "val", "label": sample["label"],
            "cache_identity_sha256": identity_sha, "metadata_sha256": sample["metadata_sha256"],
            "frame_numbers": sample["frame_numbers"], "timestamps_ms": sample["frame_timestamps_ms"],
            "input_sha256": [member["sha256"] for member in sample["members"]]}


def validate_record(record: dict, expected: dict, extractor) -> None:
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("continuous pose cache identity mismatch")
    if len(record["detections"]) != len(expected["timestamps_ms"]):
        raise ValueError("continuous cache must contain every original frame")
    associated = extractor.associate_detections(record["detections"], expected["timestamps_ms"])
    if any(record[key] != value for key, value in associated.items()):
        raise ValueError("cached association differs from causal raw-detection association")


def prepare(args, extractor, samples: dict[str, dict]) -> None:
    if extractor.sha256(POSE_CHECKPOINT) != EXPECTED_POSE_SHA256:
        raise ValueError("official pose checkpoint SHA mismatch")
    cache = args.pose_cache
    identity = {"schema_version": "1.0", "purpose": "continuous predetermined validation sequences",
                "pose_checkpoint_sha256": EXPECTED_POSE_SHA256, "settings": extractor.SETTINGS,
                "source_sha256": extractor.sha256(Path(extractor.__file__)), "device": args.device,
                "metadata_sha256": {name: sample["metadata_sha256"] for name, sample in samples.items()}}
    identity_path = cache / "cache-identity.json"
    cache.mkdir(parents=True, exist_ok=True)
    if not identity_path.exists() and any(cache.iterdir()):
        raise ValueError("refusing unrecognized nonempty continuous pose cache")
    extractor.immutable_json(identity_path, identity)
    identity_sha = extractor.sha256(identity_path)
    model = None
    record_hashes = {}
    for sequence, sample in samples.items():
        record_path = cache / f"{sequence}.json"
        expected = expected_record(sequence, sample, identity_sha)
        if record_path.exists():
            validate_record(json.loads(record_path.read_text()), expected, extractor)
            print(f"validated existing {sequence} cache", flush=True)
        else:
            if model is None:
                import torch
                from ultralytics import YOLO

                torch.set_num_threads(4)
                model = YOLO(str(POSE_CHECKPOINT))
            started = time.monotonic()
            detections = []
            # Chunk only progress reporting; detector batches and association policy stay unchanged.
            for start in range(0, len(sample["absolute_frames"]), 40):
                detections.extend(extractor.infer_frames(model, sample["absolute_frames"][start:start + 40],
                                                         args.device))
                print(f"{sequence} poses {len(detections)}/{sample['frame_count']} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
            associated = extractor.associate_detections(detections, sample["frame_timestamps_ms"])
            extractor.immutable_json(record_path, {**expected, **associated, "detections": detections})
        record_hashes[sequence] = extractor.sha256(record_path)
    extractor.immutable_json(cache / "prepared.json", {
        "schema_version": "1.0", "cache_identity_sha256": identity_sha, "record_sha256": record_hashes,
        "total_original_frames": sum(s["frame_count"] for s in samples.values()),
    })
    print(f"prepared immutable continuous pose cache: {cache}", flush=True)


def prefix_indices(prefix_length: int) -> np.ndarray:
    if prefix_length < 16:
        raise ValueError("at least 16 observed frames are required")
    return ((np.arange(16, dtype=np.int64) * 2 + 1) * prefix_length // 32).astype(np.int64)


def sampled_prefix(record: dict, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Do not bridge an identity break hidden between two sampled observations."""
    points = np.asarray(record["keypoints"], dtype=np.float32)[indices].copy()
    boxes = np.asarray(record["boxes"], dtype=np.float32)[indices].copy()
    tracking = record["tracking"]
    masked = []
    for position, (left, right) in enumerate(zip(indices[:-1], indices[1:], strict=True), start=1):
        interval = tracking[int(left):int(right) + 1]
        same_track = len({item["track_id"] for item in interval}) == 1
        if not same_track or not all(item["association_valid"] for item in interval):
            points[position, :, 2] = 0
            masked.append(position)
    return points, boxes, masked


def check(args, extractor, samples: dict[str, dict]) -> None:
    import torch
    from campus_safety_ai.training import pose_motion

    if args.checkpoint is None:
        raise ValueError("check requires --checkpoint pointing to the frozen selected best.pt")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing nonempty result directory: {args.output}")
    checkpoint = args.checkpoint.resolve()
    frozen_path, protocol_path = checkpoint.parent / "frozen.json", checkpoint.parent / "protocol.json"
    frozen, protocol = json.loads(frozen_path.read_text()), json.loads(protocol_path.read_text())
    checkpoint_sha = extractor.sha256(checkpoint)
    if frozen["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("checkpoint differs from frozen selection")
    if frozen["protocol_sha256"] != extractor.sha256(protocol_path):
        raise ValueError("protocol differs from frozen selection")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity_path = args.pose_cache / "cache-identity.json"
    identity = json.loads(identity_path.read_text())
    prepared = json.loads((args.pose_cache / "prepared.json").read_text())
    identity_sha = extractor.sha256(identity_path)
    if prepared["cache_identity_sha256"] != identity_sha:
        raise ValueError("continuous cache identity hash mismatch")
    trained_identity = protocol["pose_cache_identity"]
    for field in ("pose_checkpoint_sha256", "settings", "source_sha256"):
        if identity[field] != trained_identity[field]:
            raise ValueError(f"continuous pose extraction differs from training: {field}")
    if (payload["model_name"] != "pose-motion-classifier"
            or payload["feature_dim"] != len(pose_motion.FEATURE_NAMES)
            or payload["feature_names"] != list(pose_motion.FEATURE_NAMES)
            or payload["class_names"] != ["no_fall_transition", "fall_transition"]
            or payload["descriptor_source_sha256"] != extractor.sha256(Path(pose_motion.__file__))
            or payload["pose_checkpoint_sha256"] != identity["pose_checkpoint_sha256"]
            or payload["protocol_sha256"] != frozen["protocol_sha256"]):
        raise ValueError("classifier schema, descriptor or provenance mismatch")
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or threshold != frozen["threshold"]:
        raise ValueError("invalid or unfrozen threshold")
    model = pose_motion.PoseMotionClassifier(payload["feature_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.set_feature_normalization(model.feature_mean, model.feature_scale)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("nonfinite frozen weights")
    model.eval().requires_grad_(False).to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "1.0", "note": NOTE, "checkpoint_sha256": checkpoint_sha,
               "frozen_sha256": extractor.sha256(frozen_path), "threshold": threshold,
               "protocol_sha256": frozen["protocol_sha256"], "cache_identity_sha256": identity_sha,
               "source_sha256": extractor.sha256(Path(__file__)), "device": args.device,
               "sampling": "16 equal-bin frame-index centers over observed prefix; zero-based indices",
               "association_guard": "invalidate sampled point after any intervening invalid/change of track",
               "sequences": {}}
    for sequence, sample in samples.items():
        record_path = args.pose_cache / f"{sequence}.json"
        if extractor.sha256(record_path) != prepared["record_sha256"][sequence]:
            raise ValueError(f"continuous record changed: {record_path}")
        record = json.loads(record_path.read_text())
        validate_record(record, expected_record(sequence, sample, identity_sha), extractor)
        timeline = []
        for endpoint, timestamp in enumerate(record["timestamps_ms"]):
            row = {"frame_index": endpoint, "frame_number": record["frame_numbers"][endpoint],
                   "source_timestamp_ms": timestamp, "prefix_length": endpoint + 1,
                   "prefix_end_index": endpoint, "selected_indices": [], "score": None,
                   "prediction": "warming_up", "quality": None}
            if endpoint >= 15:
                indices = prefix_indices(endpoint + 1)
                points, boxes, masked = sampled_prefix(record, indices)
                descriptor = pose_motion.motion_descriptor(points, boxes,
                    np.asarray(record["timestamps_ms"])[indices], **payload["descriptor_settings"])
                row.update(selected_indices=indices.tolist(), quality=descriptor["quality"],
                           intervening_association_masked_positions=masked, prediction="unknown")
                if descriptor["valid"]:
                    with torch.inference_mode():
                        features = torch.from_numpy(descriptor["features"][None]).to(args.device)
                        score = float(model(features).softmax(dim=1)[0, 1].cpu())
                    if not np.isfinite(score):
                        raise ValueError("nonfinite frozen-head prediction")
                    row.update(score=score, prediction=payload["class_names"][int(score >= threshold)])
            timeline.append(row)
        first_positive = next((row["source_timestamp_ms"] for row in timeline
                               if row["prediction"] == "fall_transition"), None)
        unknown = sum(row["prediction"] == "unknown" for row in timeline)
        evaluated = len(timeline) - min(15, len(timeline))
        summary["sequences"][sequence] = {
            "sequence_label": sample["label"], "original_frame_count": len(timeline),
            "raw_pose_record_sha256": prepared["record_sha256"][sequence],
            "input_sha256": record["input_sha256"], "metadata_sha256": sample["metadata_sha256"],
            "full_prefix_final": timeline[-1], "first_positive_source_timestamp_ms": first_positive,
            "unknown_count": unknown, "warming_up_count": min(15, len(timeline)),
            "unknown_fraction_all_frames": unknown / len(timeline),
            "unknown_fraction_after_warmup": unknown / evaluated if evaluated else None,
        }
        extractor.immutable_json(args.output / f"{sequence}-timeline.json", {"note": NOTE, "frames": timeline})
        print(f"{sequence}: final={timeline[-1]['score']} prediction={timeline[-1]['prediction']} "
              f"first_positive_ms={first_positive} unknown={unknown}/{evaluated}", flush=True)
    extractor.immutable_json(args.output / "summary.json", summary)
    print(NOTE, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "check"], required=True)
    parser.add_argument("--pose-cache", type=Path, default=ROOT / "runtime/training/urfd-pose-continuous-v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/checks/urfd-pose-continuous-v1")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args()
    extractor = extractor_module()
    samples = validate_inputs(extractor)
    print("verified all 400 original PNG SHA values and official timestamps", flush=True)
    (prepare if args.phase == "prepare" else check)(args, extractor, samples)


if __name__ == "__main__":
    main()
