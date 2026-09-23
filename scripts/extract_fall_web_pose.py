#!/usr/bin/env python3
"""Extract fixed-detector poses with exactly the production single-person guard."""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from campus_safety_ai.adapters.runtimes.pose_fall import FrozenPoseFallHead, SinglePersonPose, sha256
from campus_safety_ai.adapters.timed_file_source import OpenCvTimedFileFrames
from campus_safety_ai.adapters.runtimes import pose_fall

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test", "external_diagnostic"], default=["train", "val"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/fall-web-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/training/fall-web-pose-v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runtime/training/fall-pose-dense-v1-20260921/best.pt")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/yolo26n-pose.pt")
    parser.add_argument("--detection-confidence", type=float, help="explicit new training extraction setting")
    args = parser.parse_args()
    head = FrozenPoseFallHead(args.checkpoint)
    settings = dict(head.extraction_identity["settings"])
    if args.detection_confidence is not None:
        if not 0 < args.detection_confidence <= 1:
            raise ValueError("detection confidence must be in (0,1]")
        settings["detection_confidence"] = args.detection_confidence
    # This is a new training source, not an override to a deployed frozen head.
    extraction_contract = SimpleNamespace(pose_checkpoint_sha256=sha256(args.detector),
                                          extraction_identity={"settings": settings})
    detector = SinglePersonPose(args.detector, extraction_contract, args.device)
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"pose_checkpoint_sha256": sha256(args.detector), "settings": settings,
                "runtime_source_sha256": sha256(Path(pose_fall.__file__)), "device": args.device,
                "guard": "exact SinglePersonPose.predict, every decoded frame, explicit source times"}
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("cache identity changed; use a new output directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    prepared_path = args.output / "prepared.json"
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        if prepared["identity_sha256"] != sha256(identity_path):
            raise ValueError("prepared detector identity changed")
        for name, expected in prepared["records"].items():
            if sha256(args.output / (name + ".json")) != expected:
                raise ValueError("prepared pose record was modified")
    else:
        if any(path.name != "identity.json" for path in args.output.glob("*.json")):
            raise ValueError("unregistered existing pose records require a fresh extraction directory")
        prepared = {"identity_sha256": sha256(identity_path), "records": {}}

    def save_prepared():
        partial = prepared_path.with_suffix(".json.partial")
        partial.write_text(json.dumps(prepared, indent=2) + "\n")
        partial.replace(prepared_path)

    save_prepared()
    for sample in manifest["samples"]:
        if sample["split"] not in args.splits:
            continue
        video = ROOT / sample["video_path"]
        if sha256(video) != sample["sha256"]:
            raise ValueError("source video hash mismatch")
        output = args.output / (sample["sample_id"] + ".json")
        if output.exists():
            if prepared["records"].get(sample["sample_id"]) != sha256(output):
                raise ValueError("unregistered existing pose record; do not silently adopt an orphan")
            cache = json.loads(output.read_text())
            if (cache["identity_sha256"] != sha256(identity_path) or cache["source_sha256"] != sample["sha256"]
                    or cache["timestamps_s"] != sample["timing"]["frame_timestamps_s"]):
                raise ValueError("cached pose source mismatch")
            print(json.dumps({"sample": sample["sample_id"], "status": "verified_existing"}), flush=True)
            continue
        started = time.monotonic()
        rows = []
        origin = datetime(2000, 1, 1, tzinfo=UTC)
        with OpenCvTimedFileFrames(video, sample["sample_id"], 1, origin,
                                  timestamps=sample["timing"]["frame_timestamps_s"]) as source:
            for frame in source:
                pose = detector.predict(frame)
                rows.append({"sequence": pose.sequence, "valid": pose.valid, "reason": pose.reason,
                             "track_key": pose.track_key, "keypoints": list(map(list, pose.keypoints)) if pose.valid else [],
                             "box": list(pose.box) if pose.valid else []})
        # numpy scalars must not leak into the portable cache.
        payload = {"sample_id": sample["sample_id"], "source_sha256": sample["sha256"],
                   "identity_sha256": sha256(identity_path), "timestamps_s": sample["timing"]["frame_timestamps_s"],
                   "frames": rows, "elapsed_seconds": time.monotonic() - started}
        partial = output.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, default=float, separators=(",", ":")) + "\n")
        partial.replace(output)
        prepared["records"][sample["sample_id"]] = sha256(output)
        save_prepared()
        print(json.dumps({"sample": sample["sample_id"], "frames": len(rows),
                          "valid": sum(row["valid"] for row in rows), "seconds": payload["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
