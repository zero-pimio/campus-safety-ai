"""Cache every original frame of eight fixed URFD training sequences.

Metadata is consumed only after the downloader atomically publishes a complete
sequence. Existing sequence caches are verified and reused without alteration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ("adl-01", "adl-02", "adl-04", "adl-06", "fall-01", "fall-04", "fall-05", "fall-09")
POSE_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"


def extractor_module():
    path = Path(__file__).with_name("extract_urfd_pose.py")
    spec = importlib.util.spec_from_file_location("dense_train_pose_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sequence(data_root: Path, sequence: str, extractor) -> dict:
    if sequence not in SEQUENCES:
        raise ValueError("only the eight predetermined training sequence IDs are permitted")
    metadata_path = data_root / sequence / "metadata.json"
    sample = json.loads(metadata_path.read_text())
    expected = {"sequence_id": sequence, "split": "train", "label": int(sequence.startswith("fall-")),
                "group_id": f"urfd:{sequence}"}
    if any(sample.get(key) != value for key, value in expected.items()):
        raise ValueError(f"unexpected training sequence metadata: {metadata_path}")
    count = sample.get("frame_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 16:
        raise ValueError(f"training sequence must contain at least 16 real frames: {metadata_path}")
    fields = ("frames", "members", "frame_numbers", "frame_timestamps_ms")
    if sample.get("member_count") != count or any(len(sample[key]) != count for key in fields):
        raise ValueError(f"misaligned sequence metadata: {metadata_path}")
    timestamps = np.asarray(sample["frame_timestamps_ms"], dtype=np.float64)
    numbers = sample["frame_numbers"]
    if (not np.isfinite(timestamps).all() or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError(f"invalid official timestamps: {metadata_path}")
    if (any(not isinstance(number, int) or isinstance(number, bool) or number < 1 for number in numbers)
            or any(left >= right for left, right in zip(numbers[:-1], numbers[1:], strict=True))):
        raise ValueError(f"frame numbers must be positive and strictly increasing: {metadata_path}")
    paths = []
    for name, member, number, timestamp in zip(*(sample[key] for key in fields), strict=True):
        path = (ROOT / name).resolve()
        if not path.is_relative_to((data_root / sequence).resolve()) or path.suffix.lower() != ".png":
            raise ValueError(f"unexpected training frame path: {path}")
        if (member["path"] != name or member["frame_number"] != number
                or member["timestamp_ms"] != timestamp):
            raise ValueError(f"member path/frame/timestamp mismatch: {path}")
        if path.stat().st_size != member["size"] or extractor.sha256(path) != member["sha256"]:
            raise ValueError(f"training PNG SHA/size mismatch: {path}")
        paths.append(str(path))
    if len(set(paths)) != count:
        raise ValueError(f"duplicate training PNG paths: {metadata_path}")
    return {**sample, "metadata_sha256": extractor.sha256(metadata_path), "absolute_frames": paths}


def expected_record(sample: dict, identity_sha: str, source_sha: str) -> dict:
    return {"schema_version": "1.0", "sequence_id": sample["sequence_id"], "split": "train",
            "group_id": sample["group_id"], "label": sample["label"],
            "cache_identity_sha256": identity_sha, "metadata_sha256": sample["metadata_sha256"],
            "pose_checkpoint_sha256": POSE_SHA256, "source_sha256": source_sha,
            "frame_numbers": sample["frame_numbers"], "timestamps_ms": sample["frame_timestamps_ms"],
            "input_sha256": [member["sha256"] for member in sample["members"]]}


def validate_record(record: dict, expected: dict, extractor) -> None:
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("dense training cache provenance mismatch")
    if len(record["detections"]) != len(expected["timestamps_ms"]):
        raise ValueError("dense training cache must contain every original frame")
    associated = extractor.associate_detections(record["detections"], expected["timestamps_ms"])
    if any(record.get(key) != value for key, value in associated.items()):
        raise ValueError("cached observations differ from causal raw-detection association")


def run(args) -> None:
    if not np.isfinite(args.wait_seconds) or not 0 <= args.wait_seconds <= 300:
        raise ValueError("wait-seconds must be finite and in [0, 300]")
    extractor = extractor_module()
    if extractor.sha256(args.checkpoint) != POSE_SHA256:
        raise ValueError("official frozen YOLO pose checkpoint SHA mismatch")
    source_sha = extractor.sha256(Path(extractor.__file__))
    identity = {"schema_version": "1.0", "purpose": "all original frames of eight fixed training sequences",
                "sequence_ids": list(SEQUENCES), "split": "train", "pose_checkpoint_sha256": POSE_SHA256,
                "settings": extractor.SETTINGS, "source_sha256": source_sha, "device": args.device,
                "driver_source_sha256": extractor.sha256(Path(__file__)), "data_root": str(args.data_root.resolve())}
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "cache-identity.json"
    if not identity_path.exists() and any(args.output.iterdir()):
        raise ValueError("refusing unrecognized nonempty dense training cache")
    extractor.immutable_json(identity_path, identity)
    identity_sha = extractor.sha256(identity_path)
    prepared_path = args.output / "prepared.json"
    previous = json.loads(prepared_path.read_text()) if prepared_path.exists() else None
    if previous is not None and (previous.get("cache_identity_sha256") != identity_sha
                                 or set(previous.get("record_sha256", {})) != set(SEQUENCES)):
        raise ValueError("prepared dense training cache identity or sequence membership changed")
    pending = set(SEQUENCES)
    completed, metadata_hashes, counts = {}, {}, {}
    model = None
    waiting_seconds = 0.0
    while pending:
        ready = [sequence for sequence in SEQUENCES if sequence in pending
                 and (args.data_root / sequence / "metadata.json").is_file()]
        if not ready:
            if waiting_seconds >= args.wait_seconds:
                raise TimeoutError(f"metadata still unavailable for {sorted(pending)}; rerun to resume verified cache")
            wait = min(50.0, args.wait_seconds - waiting_seconds)
            print(f"waiting up to {wait:.0f}s for complete metadata: {sorted(pending)}", flush=True)
            # Check at short intervals so a newly completed sequence starts promptly.
            started = time.monotonic()
            while time.monotonic() - started < wait:
                time.sleep(max(0.0, min(2.0, wait - (time.monotonic() - started))))
                if any((args.data_root / sequence / "metadata.json").is_file() for sequence in pending):
                    break
            waiting_seconds += time.monotonic() - started
            continue
        for sequence in ready:
            sample = validate_sequence(args.data_root, sequence, extractor)
            print(f"verified {sequence}: all {sample['frame_count']} PNG SHA values and official timestamps", flush=True)
            record_path = args.output / f"{sequence}.json"
            expected = expected_record(sample, identity_sha, source_sha)
            if record_path.exists():
                if previous is not None and extractor.sha256(record_path) != previous["record_sha256"][sequence]:
                    raise ValueError(f"prepared cache record SHA mismatch: {record_path}")
                record = json.loads(record_path.read_text())
                validate_record(record, expected, extractor)
                print(f"reused verified immutable cache: {sequence}", flush=True)
            else:
                if previous is not None:
                    raise ValueError(f"prepared cache record is missing: {record_path}")
                if model is None:
                    import torch
                    from ultralytics import YOLO

                    torch.set_num_threads(4)
                    model = YOLO(str(args.checkpoint))
                started = time.monotonic()
                detections = []
                for first in range(0, sample["frame_count"], 40):
                    detections.extend(extractor.infer_frames(model, sample["absolute_frames"][first:first + 40],
                                                             args.device))
                    print(f"{sequence} poses {len(detections)}/{sample['frame_count']} "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
                associated = extractor.associate_detections(detections, sample["frame_timestamps_ms"])
                record = {**expected, **associated, "detections": detections}
                extractor.immutable_json(record_path, record)
            completed[sequence] = extractor.sha256(record_path)
            metadata_hashes[sequence] = sample["metadata_sha256"]
            counts[sequence] = sample["frame_count"]
            pending.remove(sequence)
            print(f"completed {sequence} associated={sum(item['association_valid'] for item in record['tracking'])}"
                  f"/{sample['frame_count']} total_sequences={len(completed)}/{len(SEQUENCES)}", flush=True)
    extractor.immutable_json(prepared_path, {
        "schema_version": "1.0", "cache_identity_sha256": identity_sha, "split": "train",
        "record_sha256": {sequence: completed[sequence] for sequence in SEQUENCES},
        "metadata_sha256": {sequence: metadata_hashes[sequence] for sequence in SEQUENCES},
        "frame_counts": {sequence: counts[sequence] for sequence in SEQUENCES},
        "total_original_frames": sum(counts.values()),
    })
    print(f"prepared {len(SEQUENCES)} training sequences / {sum(counts.values())} frames: {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/private/urfd-dense-train-v1")
    parser.add_argument("--output", "--pose-cache", type=Path, default=ROOT / "runtime/training/urfd-pose-dense-train-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/yolo26n-pose.pt")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--wait-seconds", type=float, default=300.0,
                        help="Maximum cumulative idle wait for complete sequence metadata (0 to 300)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
