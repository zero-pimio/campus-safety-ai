"""Cache person-associated pose observations without fitting on video labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"image_size": 640, "detection_confidence": 0.2, "batch_size": 4,
            "tracking": "IoU association; ambiguous/disconnected frames are invalid",
            "minimum_iou": 0.05, "ambiguity_margin": 0.1, "maximum_gap_ms": 2000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"refusing to overwrite differing cache: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def intersection_over_union(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right - left, 0).prod(1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    return intersection / np.maximum(area + areas - intersection, 1e-8)


def associate_detections(detections: list[dict], timestamps_ms: list[float]) -> dict:
    times = np.asarray(timestamps_ms, dtype=np.float64)
    if (times.shape != (len(detections),) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("detections need aligned, finite, nonnegative and strictly increasing timestamps")
    selected_keypoints, selected_boxes, tracking = [], [], []
    anchor, anchor_time, track_id = None, None, 0
    for index, (detection, timestamp) in enumerate(zip(detections, timestamps_ms, strict=True)):
        boxes = np.asarray(detection["boxes"], dtype=np.float32).reshape(-1, 4)
        points = np.asarray(detection["keypoints"], dtype=np.float32).reshape(-1, 17, 3)
        scores = np.asarray(detection["scores"], dtype=np.float32)
        if len(boxes) != len(points) or len(boxes) != len(scores):
            raise ValueError("misaligned pose detections")
        selected, status, valid = None, "missing", False
        if len(boxes):
            if not (np.isfinite(boxes).all() and np.isfinite(points).all() and np.isfinite(scores).all()):
                raise ValueError("nonfinite pose detections")
            if anchor is None or timestamp - anchor_time > SETTINGS["maximum_gap_ms"]:
                areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
                selected = int(np.argmax(scores * np.sqrt(areas)))
                track_id += 1
                # A new identity is not a motion observation relative to the preceding frame.
                valid = index == 0 and len(boxes) == 1
                status = "initial_single_person" if valid else "new_anchor_invalid"
                anchor, anchor_time = boxes[selected].copy(), timestamp
            else:
                overlaps = intersection_over_union(anchor, boxes)
                order = np.argsort(overlaps)[::-1]
                selected = int(order[0])
                ambiguous = len(order) > 1 and overlaps[order[1]] > SETTINGS["minimum_iou"] and (
                    overlaps[order[0]] - overlaps[order[1]] < SETTINGS["ambiguity_margin"])
                if overlaps[selected] < SETTINGS["minimum_iou"]:
                    track_id += 1
                    status = "disconnected_anchor_invalid"
                    anchor, anchor_time = boxes[selected].copy(), timestamp
                elif ambiguous:
                    status = "ambiguous_invalid"
                else:
                    valid, status = True, "associated"
                    anchor, anchor_time = boxes[selected].copy(), timestamp
        if selected is None or not valid:
            selected_keypoints.append(np.zeros((17, 3), dtype=np.float32).tolist())
            selected_boxes.append(np.zeros(4, dtype=np.float32).tolist())
        else:
            selected_keypoints.append(points[selected].tolist())
            selected_boxes.append(boxes[selected].tolist())
        tracking.append({"frame_index": index, "detection_count": len(boxes), "selected_index": selected,
                         "track_id": track_id if selected is not None else None,
                         "association_valid": valid, "status": status})
    return {"keypoints": selected_keypoints, "boxes": selected_boxes, "tracking": tracking}


def infer_frames(model, paths: list[str], device: str) -> list[dict]:
    detections = []
    for start in range(0, len(paths), SETTINGS["batch_size"]):
        results = model.predict(paths[start:start + SETTINGS["batch_size"]],
                                device=device, imgsz=SETTINGS["image_size"],
                                conf=SETTINGS["detection_confidence"], verbose=False, save=False)
        for result in results:
            detections.append({"boxes": result.boxes.xyxy.cpu().tolist(),
                               "scores": result.boxes.conf.cpu().tolist(),
                               "keypoints": result.keypoints.data.cpu().tolist()})
    if len(detections) != len(paths):
        raise ValueError("pose predictor did not return every input frame")
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/urfd-sequences-v1.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/yolo26n-pose.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/training/urfd-pose-cache-v1")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("download the official pose checkpoint before extraction")
    spec = importlib.util.spec_from_file_location("urfd_manifest_validation", Path(__file__).with_name("train_urfd_baseline.py"))
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    manifest = json.loads(args.manifest.read_text())
    samples = baseline.validate_manifest(manifest, ROOT)
    manifest_sha, checkpoint_sha = sha256(args.manifest), sha256(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"schema_version": "1.0", "manifest_sha256": manifest_sha,
                "pose_checkpoint_sha256": checkpoint_sha, "settings": SETTINGS,
                "source_sha256": sha256(Path(__file__)), "device": args.device}
    descriptor_path = args.output / "cache-identity.json"
    if not descriptor_path.exists() and any(args.output.iterdir()):
        raise ValueError("refusing an unrecognized nonempty pose cache")
    immutable_json(descriptor_path, identity)
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(4)
    model = YOLO(str(args.checkpoint))
    selected = [s for s in samples if s["split"] in args.splits]
    start = time.monotonic()
    for index, sample in enumerate(selected):
        path = args.output / f"{sample['sequence_id']}.json"
        expected = {"sequence_id": sample["sequence_id"], "label": sample["label"], "split": sample["split"],
                    "schema_version": "1.0", "manifest_sha256": manifest_sha,
                    "pose_checkpoint_sha256": checkpoint_sha, "timestamps_ms": sample["frame_timestamps_ms"],
                    "frame_numbers": sample["frame_numbers"], "input_sha256": [m["sha256"] for m in sample["members"]],
                    "cache_identity_sha256": sha256(descriptor_path)}
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"cached observation identity mismatch: {path}")
            print(f"cached {sample['sequence_id']}", flush=True)
            continue
        detections = infer_frames(model, [str(ROOT / name) for name in sample["frames"]], args.device)
        associated = associate_detections(detections, sample["frame_timestamps_ms"])
        record = {**expected, **associated, "detections": detections}
        immutable_json(path, record)
        print(f"pose {index + 1}/{len(selected)} {sample['sequence_id']} {sample['split']} "
              f"associated={sum(t['association_valid'] for t in associated['tracking'])}/16 "
              f"elapsed={time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
