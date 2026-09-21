"""Train with actual dense causal prefixes from eight prespecified training videos.

Posture annotations are a conservative proxy for pre-transition observations,
not verified RGB event boundaries. The existing 14-sequence test is reused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from campus_safety_ai.training.pose_motion import FEATURE_NAMES, motion_descriptor

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("pose_motion_training_helpers", Path(__file__).with_name("train_urfd_pose_motion.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def negative_prefixes(samples: list[dict], cache: Path, annotations: dict) -> tuple[list[dict], list[dict]]:
    annotated = {row["sequence_id"]: row for row in annotations["fall_sequences"]}
    records, audit = [], []
    for sample in samples:
        if sample["split"] != "train":
            raise ValueError("derived training prefixes must come only from train groups")
        raw = json.loads((cache / f"{sample['sequence_id']}.json").read_text())
        points, boxes, times = np.asarray(raw["keypoints"]), np.asarray(raw["boxes"]), np.asarray(raw["timestamps_ms"])
        maximum = 15  # keep the original 16-frame sequence as its own observation
        basis = "original ADL sequence category"
        if sample["label"] == 1:
            annotation = annotated[sample["sequence_id"]]
            if annotation["group_id"] != sample["group_id"] or annotation["split"] != "train":
                raise ValueError("annotation group or split identity mismatch")
            if annotation["sync_sha256"] != sample["sync_sha256"]:
                raise ValueError("annotation synchronization identity mismatch")
            rows = annotation["sampled_frames"]
            if [r["frame_number"] for r in rows] != sample["frame_numbers"]:
                raise ValueError("annotation frame correspondence mismatch")
            if [r["timestamp_ms"] for r in rows] != sample["frame_timestamps_ms"]:
                raise ValueError("annotation timestamps differ from the source manifest")
            count = 0
            for row in rows:
                if not row["before_first_zero"] or row["posture_label"] != -1:
                    break
                count += 1
            maximum = min(maximum, count - 1)  # one actual sampled frame of margin before the annotated transition
            basis = "all observed frames have posture -1 before first 0; one sampled-frame margin; proxy negative"
        added = 0
        for length in range(4, maximum + 1):
            descriptor = motion_descriptor(points[:length], boxes[:length], times[:length])
            row = {"sequence_id": f"{sample['sequence_id']}:prefix-{length}", "parent_sequence_id": sample["sequence_id"],
                   "group_id": sample["group_id"], "label": 0, "split": "train", "derived": True,
                   "label_basis": basis, "original_sequence_label": sample["label"],
                   "prefix_length": length, "end_frame_number": sample["frame_numbers"][length - 1],
                   "end_timestamp_ms": float(times[length - 1]), "valid": bool(descriptor["valid"]),
                   "quality": descriptor["quality"],
                   "features": descriptor["features"].tolist() if descriptor["valid"] else None}
            records.append(row)
            added += bool(descriptor["valid"])
        audit.append({"sequence_id": sample["sequence_id"], "original_label": sample["label"],
                      "maximum_prefix_length": maximum, "valid_derived_prefixes": added, "label_basis": basis})
    return records, audit


DENSE_IDS = ("adl-01", "adl-02", "adl-04", "adl-06", "fall-01", "fall-04", "fall-05", "fall-09")


def dense_prefixes(samples: list[dict], cache: Path, annotations: dict) -> tuple[list[dict], list[dict]]:
    spec = importlib.util.spec_from_file_location("causal_sampling_helpers", Path(__file__).with_name("check_urfd_pose_continuous.py"))
    causal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(causal)
    by_id = {s["sequence_id"]: s for s in samples}
    annotated = {r["sequence_id"]: r for r in annotations["fall_sequences"]}
    records, audit = [], []
    for sequence in DENSE_IDS:
        sample = by_id[sequence]
        if sample["split"] != "train":
            raise ValueError("dense training may not read validation or test groups")
        raw = json.loads((cache / f"{sequence}.json").read_text())
        if (raw["sequence_id"] != sequence or raw["split"] != "train"
                or raw["label"] != sample["label"] or raw["group_id"] != sample["group_id"]):
            raise ValueError("dense cache sequence identity mismatch")
        times = np.asarray(raw["timestamps_ms"])
        numbers = np.asarray(raw["frame_numbers"])
        if (times.ndim != 1 or len(times) < 16 or times.shape != numbers.shape
                or np.any(np.diff(times) <= 0) or not np.isfinite(times).all()
                or np.any(np.diff(numbers) <= 0) or times[0] < 0):
            raise ValueError("invalid dense observation timestamps")
        first_zero, first_one = None, None
        if sample["label"] == 1:
            annotation = annotated[sequence]
            if annotation["split"] != "train" or annotation["group_id"] != sample["group_id"] or annotation["sync_sha256"] != sample["sync_sha256"]:
                raise ValueError("dense annotation identity mismatch")
            first_zero = next(r["first_timestamp_ms"] for r in annotation["posture_intervals"] if r["posture_label"] == 0)
            first_one = next(r["first_timestamp_ms"] for r in annotation["posture_intervals"] if r["posture_label"] == 1)
        skipped = 0
        for length in sorted({*range(16, len(times) + 1, 3), len(times)}):
            indices = causal.prefix_indices(length)
            latest_input_time = times[indices[-1]]
            label, basis = 0, "official ADL source category"
            if sample["label"] == 1:
                if latest_input_time < first_zero - 200:
                    label, basis = 0, "pre-transition posture proxy with 200ms margin"
                elif latest_input_time >= first_one + 200:
                    label, basis = 1, "original fall sequence with completed-posture proxy plus 200ms margin"
                else:
                    skipped += 1
                    continue
            points, boxes, masked = causal.sampled_prefix(raw, indices)
            descriptor = motion_descriptor(points, boxes, times[indices])
            records.append({"sequence_id": f"{sequence}:dense-prefix-{length}", "parent_sequence_id": sequence,
                            "group_id": sample["group_id"], "label": label, "split": "train", "derived": True,
                            "label_basis": basis, "original_sequence_label": sample["label"],
                            "prefix_length": length, "selected_indices": indices.tolist(),
                            "latest_input_frame_number": int(numbers[indices[-1]]),
                            "latest_input_timestamp_ms": float(latest_input_time),
                            "intervening_association_masked_positions": masked,
                            "valid": bool(descriptor["valid"]), "quality": descriptor["quality"],
                            "features": descriptor["features"].tolist() if descriptor["valid"] else None})
        audit.append({"sequence_id": sequence, "source_frame_count": len(times),
                      "uncertain_transition_prefixes_skipped": skipped,
                      "first_posture_zero_ms": first_zero, "first_posture_one_ms": first_one})
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-cache", type=Path, default=ROOT / "runtime/training/urfd-pose-cache-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-pose-cache", type=Path, default=ROOT / "runtime/training/urfd-pose-dense-train-v1")
    args = parser.parse_args()
    h = helper()
    torch.set_num_threads(4)
    torch.manual_seed(h.SEED)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing nonempty follow-up experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "datasets/manifests/urfd-sequences-v1.json"
    manifest, manifest_sha = h.read_manifest(manifest_path)
    identity = h.read_identity(args.pose_cache, manifest_sha)
    samples = [s for s in manifest["samples"] if s["split"] in ("train", "val")]
    records, controls, fingerprints = h.read_cohort(samples, args.pose_cache, manifest_sha, identity)
    annotation_path = ROOT / "datasets/private/urfd-annotations-v1/audit.json"
    annotations = json.loads(annotation_path.read_text())
    if annotations.get("manifest_sha256") != manifest_sha:
        raise ValueError("annotation audit was generated from a different manifest")
    dense_identity = json.loads((args.dense_pose_cache / "cache-identity.json").read_text())
    for key in ("source_sha256", "pose_checkpoint_sha256", "settings"):
        if dense_identity[key] != identity[key]:
            raise ValueError(f"dense extractor differs from sparse training: {key}")
    prepared = json.loads((args.dense_pose_cache / "prepared.json").read_text())
    dense_identity_sha = h.sha256(args.dense_pose_cache / "cache-identity.json")
    if (prepared["cache_identity_sha256"] != dense_identity_sha or prepared["split"] != "train"
            or dense_identity["split"] != "train" or dense_identity["sequence_ids"] != list(DENSE_IDS)
            or dense_identity["driver_source_sha256"] != h.sha256(ROOT / "scripts/extract_urfd_dense_train_pose.py")):
        raise ValueError("dense prepared identity or source mismatch")
    dense_hashes = {name: h.sha256(args.dense_pose_cache / f"{name}.json") for name in DENSE_IDS}
    if prepared["record_sha256"] != dense_hashes:
        raise ValueError("dense pose records differ from verified prepared cache")
    by_id = {sample["sequence_id"]: sample for sample in samples}
    for sequence in DENSE_IDS:
        raw = json.loads((args.dense_pose_cache / f"{sequence}.json").read_text())
        metadata_path = Path(dense_identity["data_root"]) / sequence / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        sample = by_id[sequence]
        if (raw["cache_identity_sha256"] != dense_identity_sha
                or raw["metadata_sha256"] != h.sha256(metadata_path)
                or raw["metadata_sha256"] != prepared["metadata_sha256"][sequence]
                or metadata["original_manifest_sha256"] != manifest_sha
                or metadata["sync_sha256"] != sample["sync_sha256"]
                or any(metadata[key] != sample[key] for key in ("sequence_id", "group_id", "split", "label"))
                or raw["timestamps_ms"] != metadata["frame_timestamps_ms"]
                or raw["frame_numbers"] != metadata["frame_numbers"]
                or prepared["frame_counts"][sequence] != len(raw["timestamps_ms"])):
            raise ValueError(f"dense metadata or synchronization mismatch: {sequence}")
    sources = {name: h.sha256(ROOT / name) for name in (*h.SOURCE_FILES, "scripts/train_urfd_pose_dense.py",
                                                      "scripts/check_urfd_pose_continuous.py",
                                                      "scripts/prepare_urfd_dense_train.py",
                                                      "scripts/extract_urfd_dense_train_pose.py")}
    protocol = {
        "schema_version": "1.0", "created_at": datetime.now(UTC).isoformat(), "seed": h.SEED,
        "experiment": "fixed dense-frame follow-up for short-prefix distribution mismatch; no test-based selection",
        "manifest_sha256": manifest_sha, "pose_cache_identity": identity,
        "pose_cache_identity_sha256": h.sha256(args.pose_cache / "cache-identity.json"),
        "development_cache_sha256": fingerprints, "source_sha256": sources,
        "class_names": h.CLASS_NAMES, "feature_names": list(FEATURE_NAMES), "descriptor_settings": {"min_conf": h.MIN_CONF},
        "annotation_audit_sha256": h.sha256(annotation_path),
        "dense_pose_cache_sha256": dense_hashes, "dense_cache_identity": dense_identity,
        "dense_metadata_sha256": prepared["metadata_sha256"],
        "dense_prepared_sha256": h.sha256(args.dense_pose_cache / "prepared.json"),
        "dense_sampling": "8 prespecified train videos; actual prefixes >=16 frames, every3 endpoints; same causal 16-bin centers and track-gap guard as continuous check",
        "dense_labels": "ADL always0; original fall before first posture0 minus200ms=>0, after first posture1 plus200ms=>1; ambiguous transition prefixes excluded",
        "dense_label_limitation": "depth-posture temporal proxy, not verified RGB event truth",
        "augmentation": "train only: ADL prefixes length 4..15; fall prefixes entirely -1 before first 0, excluding last eligible sampled frame as margin",
        "augmentation_labels": "ADL source label; pre-transition depth-posture proxy, not verified RGB event boundary",
        "normalization": "fit original valid 42 train sequences only; population std floor 1e-6",
        "loss": "class-balanced CE(original plus valid derived train prefixes) + 0.25 CE(single raw zero,0)",
        "sampling_weight": "each valid training row weighted equally before binary class balancing; derived rows are correlated, not new independent videos",
        "candidates": [0, 16], "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.01},
        "max_epochs": h.MAX_EPOCHS, "patience": h.PATIENCE,
        "selection": "epoch by unweighted original real val CE; threshold/candidate by original val BA,F1,loss with static-zero gate",
        "reused_test_split": True, "test_policy": "use existing frozen test-phase helper once after selection; not a new holdout",
        "limits": ["proxy annotation, unknown RGB boundary alignment", "derived prefixes not independent videos",
                   "same known public split reused", "no campus or subject-independent evidence"],
    }
    h.write_json(args.output / "protocol.json", protocol)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    derived, audit = negative_prefixes([s for s in samples if s["split"] == "train"], args.pose_cache, annotations)
    dense_records, dense_audit = dense_prefixes([s for s in samples if s["split"] == "train"], args.dense_pose_cache, annotations)
    h.write_json(args.output / "dense-prefixes.json", {"independent_dense_train_groups": len(DENSE_IDS),
                                                    "valid_derived_count": sum(r["valid"] for r in dense_records),
                                                    "audit": dense_audit, "records": dense_records})
    derived.extend(dense_records)
    h.write_json(args.output / "derived-prefixes.json", {"original_train_count": len(train), "independent_group_count": len({r['group_id'] for r in train}),
                                                       "valid_derived_count": sum(r["valid"] for r in derived), "audit": audit, "records": derived})
    h.write_json(args.output / "development-descriptors.json", {"records": records, "feature_names": list(FEATURE_NAMES)})
    features, _ = h.valid_tensors(train)
    mean, scale = features.mean(0), features.std(0, unbiased=False).clamp_min(h.SCALE_FLOOR)
    candidates = [h.train_candidate(hidden, train + derived, val, controls, mean, scale, args.output) for hidden in (0, 16)]
    eligible = [(result, model) for result, model in candidates if result["threshold_selection"]["eligible"]]
    if not eligible:
        raise ValueError("no eligible model; test remains unused for this experiment")
    def rank(item):
        m = item[0]["validation"]["metrics"]
        return m["balanced_accuracy"], m["f1"], -m["loss"], -item[0]["parameter_count"]
    winner, model = max(eligible, key=rank)
    threshold = winner["threshold_selection"]["threshold"]
    checkpoint = {"model_name": "pose-motion-classifier", "model_state": model.state_dict(),
                  "feature_dim": len(FEATURE_NAMES), "hidden_dim": winner["hidden_dim"], "feature_names": list(FEATURE_NAMES),
                  "class_names": h.CLASS_NAMES, "threshold": threshold, "manifest_sha256": manifest_sha,
                  "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"], "protocol_sha256": h.sha256(args.output / "protocol.json"),
                  "descriptor_source_sha256": sources["src/campus_safety_ai/training/pose_motion.py"],
                  "descriptor_settings": {"min_conf": h.MIN_CONF}, "best_epoch": winner["best_epoch"]}
    torch.save(checkpoint, args.output / "best.pt")
    checkpoint_sha = h.sha256(args.output / "best.pt")
    validation = {**winner["validation"], "split": "val", "checkpoint_sha256": checkpoint_sha,
                  "manifest_sha256": manifest_sha, "threshold_selection": winner["threshold_selection"]}
    h.write_json(args.output / "validation.json", validation)
    h.write_json(args.output / "static-gate.json", winner["static_gate"])
    h.export_head(model, val, args.output, checkpoint_sha, threshold)
    artifacts = {p.name: h.sha256(p) for p in args.output.iterdir() if p.is_file()}
    frozen = {"frozen_at": datetime.now(UTC).isoformat(), "checkpoint_sha256": checkpoint_sha,
              "validation_sha256": h.sha256(args.output / "validation.json"), "protocol_sha256": checkpoint["protocol_sha256"],
              "export_sha256": h.sha256(args.output / "export.json"), "manifest_sha256": manifest_sha,
              "descriptor_source_sha256": checkpoint["descriptor_source_sha256"], "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"],
              "source_sha256": sources, "artifact_sha256": artifacts, "hidden_dim": winner["hidden_dim"],
              "threshold": threshold, "best_epoch": winner["best_epoch"], "reused_test_split": True}
    h.write_json(args.output / "frozen.json", frozen)
    print(json.dumps({"output": str(args.output), "hidden_dim": winner["hidden_dim"], "threshold": threshold,
                      "val": validation["metrics"], "valid_derived_train_prefixes": sum(r["valid"] for r in derived)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
