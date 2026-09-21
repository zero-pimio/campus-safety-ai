"""Train a pose-motion research baseline; evaluate the reused test split separately.

The train phase never opens test pose caches. Unknown observations are retained
in reports, never imputed as stationary negatives. ONNX exports only the learned
32-feature classification head, not YOLO or the motion descriptor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from campus_safety_ai.training.calibration import predictions_at_threshold
from campus_safety_ai.training.metrics import classification_metrics, confusion_matrix
from campus_safety_ai.training.pose_motion import FEATURE_NAMES, PoseMotionClassifier, motion_descriptor

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922
SPLIT_SEED = 20260921
MAX_EPOCHS = 300
PATIENCE = 40
MIN_CONF = 0.25
SCALE_FLOOR = 1e-6
CLASS_NAMES = ["no_fall_transition", "fall_transition"]
SOURCE_FILES = (
    "scripts/train_urfd_pose_motion.py", "scripts/extract_urfd_pose.py",
    "src/campus_safety_ai/training/pose_motion.py",
    "src/campus_safety_ai/training/metrics.py", "src/campus_safety_ai/training/calibration.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    # Every artifact is new. Failed/running experiment directories are not reused.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("dataset") != "urfd-cam0" or manifest.get("seed") != SPLIT_SEED
            or manifest.get("frame_count") != 16 or manifest.get("class_names") != ["non_fall", "fall"]):
        raise ValueError("unexpected URFD manifest protocol")
    planned = {}
    for prefix, count, label, train, val in (("fall", 30, 1, 18, 6), ("adl", 40, 0, 24, 8)):
        names = [f"{prefix}-{index:02d}" for index in range(1, count + 1)]
        random.Random(SPLIT_SEED).shuffle(names)
        for index, name in enumerate(names):
            planned[name] = (label, "train" if index < train else "val" if index < train + val else "test")
    seen = set()
    for sample in manifest["samples"]:
        name = sample["sequence_id"]
        if (name in seen or planned.get(name) != (sample["label"], sample["split"])
                or sample["group_id"] != f"urfd:{name}"):
            raise ValueError("manifest differs from the fixed sequence-disjoint 42/14/14 split")
        seen.add(name)
        times = np.asarray(sample["frame_timestamps_ms"], dtype=np.float64)
        if times.shape != (16,) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
            raise ValueError("manifest timestamps must contain 16 increasing nonnegative values")
    if seen != set(planned):
        raise ValueError("manifest must contain all 70 prespecified sequences")
    return manifest, hashlib.sha256(raw).hexdigest()


def read_identity(cache: Path, manifest_sha: str) -> dict:
    identity = json.loads((cache / "cache-identity.json").read_text())
    if identity.get("schema_version") != "1.0" or identity.get("manifest_sha256") != manifest_sha:
        raise ValueError("pose cache identity does not match manifest")
    digest = identity.get("pose_checkpoint_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid pose checkpoint SHA256")
    if identity.get("source_sha256") != sha256(ROOT / "scripts/extract_urfd_pose.py"):
        raise ValueError("pose extractor source differs from the cache identity")
    return identity


def read_cohort(samples: list[dict], cache: Path, manifest_sha: str, identity: dict,
                expected_hashes: dict[str, str] | None = None) -> tuple[list[dict], list[dict], dict[str, str]]:
    records, controls, fingerprints = [], [], {}
    identity_sha = sha256(cache / "cache-identity.json")
    for sample in samples:
        name = sample["sequence_id"]
        path = cache / f"{name}.json"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        fingerprints[name] = digest
        if expected_hashes is not None and expected_hashes.get(name) != digest:
            raise ValueError(f"pose cache changed after protocol was written: {name}")
        payload = json.loads(raw)
        if payload.get("cache_identity_sha256") != identity_sha:
            raise ValueError(f"pose cache {name}: cache identity SHA mismatch")
        for key, expected in (("schema_version", "1.0"), ("manifest_sha256", manifest_sha),
                              ("pose_checkpoint_sha256", identity["pose_checkpoint_sha256"]),
                              ("sequence_id", name), ("label", sample["label"]), ("split", sample["split"])):
            if payload.get(key) != expected:
                raise ValueError(f"pose cache {name}: {key} mismatch")
        points = np.asarray(payload["keypoints"], dtype=np.float64)
        boxes = np.asarray(payload["boxes"], dtype=np.float64)
        times = np.asarray(payload["timestamps_ms"], dtype=np.float64)
        if points.shape != (16, 17, 3) or boxes.shape != (16, 4):
            raise ValueError(f"pose cache {name} has unexpected shape")
        if not np.array_equal(times, np.asarray(sample["frame_timestamps_ms"], dtype=np.float64)):
            raise ValueError(f"pose cache {name} timestamps differ from source manifest")
        if ("members" in sample and payload.get("input_sha256") != [member["sha256"] for member in sample["members"]]
                or "frame_numbers" in sample and payload.get("frame_numbers") != sample["frame_numbers"]):
            raise ValueError(f"pose cache {name} input frame identity mismatch")
        result = motion_descriptor(points, boxes, times, min_conf=MIN_CONF)
        if result["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("descriptor feature ordering changed")
        if result["valid"] and not np.isfinite(result["features"]).all():
            raise ValueError("valid descriptor must be finite")
        record = {"sequence_id": name, "group_id": sample["group_id"], "label": sample["label"],
                  "split": sample["split"], "valid": bool(result["valid"]), "quality": result["quality"],
                  "tracking": payload.get("tracking", {}),
                  "features": result["features"].tolist() if result["valid"] else None}
        records.append(record)
        # Validation controls are paired diagnostics, not additional real cases.
        if sample["split"] == "val":
            for endpoint, index in (("first", 0), ("last", -1)):
                repeated = motion_descriptor(np.repeat(points[index:index + 1] if index == 0 else points[-1:], 16, 0),
                                             np.repeat(boxes[index:index + 1] if index == 0 else boxes[-1:], 16, 0),
                                             times, min_conf=MIN_CONF)
                if repeated["valid"] and np.any(repeated["features"] != 0):
                    raise ValueError("a valid repeated static pose must produce exactly zero motion")
                controls.append({"sequence_id": name, "endpoint": endpoint,
                                 "valid": bool(repeated["valid"]), "quality": repeated["quality"]})
    return records, controls, fingerprints


def valid_tensors(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    valid = [record for record in records if record["valid"]]
    features = torch.tensor([record["features"] for record in valid], dtype=torch.float32).reshape(-1, len(FEATURE_NAMES))
    labels = torch.tensor([record["label"] for record in valid], dtype=torch.long)
    return features, labels


def coverage(records: list[dict]) -> dict:
    count = sum(record["valid"] for record in records)
    return {"available_sample_count": len(records), "valid_sample_count": count,
            "unknown_sample_count": len(records) - count, "coverage": count / len(records) if records else 0.0,
            "by_label": {str(label): {"available": sum(r["label"] == label for r in records),
                                      "valid": sum(r["label"] == label and r["valid"] for r in records)}
                         for label in (0, 1)}}


def evaluate(model: PoseMotionClassifier, records: list[dict], threshold: float) -> dict:
    features, labels = valid_tensors(records)
    model.eval()
    scores = []
    loss = None
    if len(labels):
        with torch.inference_mode():
            logits = model(features)
            loss = float(torch.nn.functional.cross_entropy(logits, labels))
            scores = logits.softmax(1)[:, 1].tolist()
    predictions = predictions_at_threshold(scores, threshold)
    rows, index = [], 0
    for record in records:
        row = {key: record[key] for key in ("sequence_id", "group_id", "label", "quality")}
        row.update(status="unknown", fall_score=None, prediction=None)
        if record["valid"]:
            row.update(status="valid", fall_score=scores[index], prediction=predictions[index])
            index += 1
        rows.append(row)
    return {**coverage(records), "sample_count": len(labels), "partial": len(labels) != len(records),
            "threshold": threshold, "metrics_scope": "valid real sequences only; unknowns excluded explicitly",
            "metrics": asdict(classification_metrics(loss, labels.tolist(), predictions)) if len(labels) else None,
            "confusion_matrix": confusion_matrix(labels.tolist(), predictions), "predictions": rows}


def select_threshold(labels: list[int], scores: list[float], static_score: float) -> dict:
    """Optimize real validation metrics, with a separate strict static gate."""
    if len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("threshold selection needs valid real validation examples from both classes")
    predictions_at_threshold([*scores, static_score])
    unique = sorted(set(scores))
    # A float32-representable successor also preserves the strict gate for float32 consumers.
    after_static = float(np.nextafter(np.float32(static_score), np.float32(math.inf)))
    candidates = {0.0, 0.5, 1.0, after_static,
                  *(left + (right - left) / 2 for left, right in zip(unique[:-1], unique[1:], strict=True))}
    eligible = [threshold for threshold in candidates if 0 <= threshold <= 1 and static_score < threshold]
    if not eligible:
        return {"eligible": False, "reason": "no legal threshold rejects the synthetic static zero",
                "static_score": static_score, "candidate_count": 0}
    ranked = [(threshold, classification_metrics(0, labels, predictions_at_threshold(scores, threshold)))
              for threshold in eligible]
    threshold, metrics = max(ranked, key=lambda item: (item[1].balanced_accuracy, item[1].f1,
                                                     -abs(item[0] - 0.5), item[0]))
    return {"eligible": True, "threshold": threshold, "static_score": static_score,
            "static_false_positive_count": 0, "candidate_count": len(eligible),
            "objective": "real validation balanced_accuracy, f1, closest_to_0.5, higher_threshold",
            "constraint": "static_score < threshold; prediction uses score >= threshold",
            "real_validation_sample_count": len(labels),
            "real_metrics": {key: value for key, value in asdict(metrics).items() if key != "loss"}}


def static_report(controls: list[dict], score: float, threshold: float) -> dict:
    valid = sum(row["valid"] for row in controls)
    prediction = int(score >= threshold)
    return {"canonical_raw_zero_score": score, "canonical_raw_zero_prediction": prediction,
            "eligible": prediction == 0, "threshold": threshold,
            "available_control_count": len(controls), "valid_control_count": valid,
            "unknown_control_count": len(controls) - valid, "false_positive_count": valid * prediction,
            "unique_valid_descriptor_count": int(valid > 0),
            "real_sample_count": 0, "interpretation": "synthetic behavior constraint, not real negative generalization",
            "controls": [{**row, "status": "valid" if row["valid"] else "unknown",
                          "fall_score": score if row["valid"] else None,
                          "prediction": prediction if row["valid"] else None} for row in controls]}


def parameter_delta(initial: dict, final: dict) -> dict:
    difference = torch.cat([(final[key] - initial[key]).reshape(-1) for key in initial])
    return {"parameter_count": difference.numel(), "changed_parameter_count": int((difference != 0).sum()),
            "maximum_absolute_change": float(difference.abs().max()), "l2_change": float(difference.norm())}


def train_candidate(hidden: int, train: list[dict], val: list[dict], controls: list[dict],
                    mean: torch.Tensor, scale: torch.Tensor, output: Path) -> tuple[dict, PoseMotionClassifier]:
    torch.manual_seed(SEED)
    model = PoseMotionClassifier(len(FEATURE_NAMES), hidden)
    model.set_feature_normalization(mean, scale)
    initial = copy.deepcopy(model.head.state_dict())
    torch.save(initial, output / f"candidate-hidden-{hidden}-initial-head.pt")
    features, labels = valid_tensors(train)
    class_weights = len(labels) / (2 * torch.bincount(labels, minlength=2).float())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01, weight_decay=0.01)
    zero = torch.zeros(1, len(FEATURE_NAMES))
    zero_label = torch.zeros(1, dtype=torch.long)
    best_loss, best_epoch, best_head = math.inf, 0, None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        real_loss = criterion(model(features), labels)
        static_loss = torch.nn.functional.cross_entropy(model(zero), zero_label)
        loss = real_loss + 0.25 * static_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        validation = evaluate(model, val, 0.5)
        val_loss = validation["metrics"]["loss"]
        history.append({"epoch": epoch, "real_weighted_ce": float(real_loss.detach()),
                        "synthetic_zero_ce": float(static_loss.detach()), "total_loss": float(loss.detach()),
                        "validation_real_ce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_head = val_loss, epoch, copy.deepcopy(model.head.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            print(f"hidden={hidden} epoch={epoch} real_ce={float(real_loss.detach()):.5f} "
                  f"zero_ce={float(static_loss.detach()):.5f} val_ce={val_loss:.5f}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    model.head.load_state_dict(best_head)
    model.eval()
    validation = evaluate(model, val, 0.5)
    with torch.inference_mode():
        static_score = float(model(zero).softmax(1)[0, 1])
    rows = [row for row in validation["predictions"] if row["status"] == "valid"]
    selection = select_threshold([r["label"] for r in rows], [r["fall_score"] for r in rows], static_score)
    result = {"hidden_dim": hidden, "parameter_count": sum(p.numel() for p in model.head.parameters()),
              "best_epoch": best_epoch, "epochs_completed": len(history), "history": history,
              "parameter_delta": parameter_delta(initial, model.head.state_dict()),
              "threshold_selection": selection, "validation_at_0_5": validation}
    if selection["eligible"]:
        result["validation"] = evaluate(model, val, selection["threshold"])
        result["static_gate"] = static_report(controls, static_score, selection["threshold"])
    torch.save(model.head.state_dict(), output / f"candidate-hidden-{hidden}-head.pt")
    write_json(output / f"candidate-hidden-{hidden}.json", result)
    return result, model


def export_head(model: PoseMotionClassifier, val: list[dict], output: Path, checkpoint_sha: str,
                threshold: float) -> dict:
    import onnx
    import onnxruntime as ort

    features, _ = valid_tensors(val)
    path = output / "model.onnx"
    torch.onnx.export(model.eval(), features[:1], path, input_names=["motion_features"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    with torch.inference_mode():
        for row in features:
            expected = model(row[None]).numpy()
            actual = session.run(None, {"motion_features": row[None].numpy()})[0]
            errors.append(float(np.abs(expected - actual).max()))
    if not all(math.isfinite(error) and error <= 1e-4 for error in errors):
        raise ValueError("classifier ONNX parity failed")
    result = {"onnx_sha256": sha256(path), "checkpoint_sha256": checkpoint_sha,
              "input_shape": [1, len(FEATURE_NAMES)], "input": "raw, unstandardized motion descriptor",
              "class_names": CLASS_NAMES, "threshold": threshold, "validation_sample_count": len(features),
              "maximum_logit_error": max(errors),
              "scope": "classification head with train-only normalization; excludes YOLO and descriptor"}
    write_json(output / "export.json", result)
    return result


def train_phase(manifest_path: Path, cache: Path, output: Path) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing a nonempty experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = read_manifest(manifest_path)
    identity = read_identity(cache, manifest_sha)
    samples = [sample for sample in manifest["samples"] if sample["split"] in ("train", "val")]
    # Fingerprint only the 56 development caches. No test cache is opened/stat'ed.
    fingerprints = {sample["sequence_id"]: sha256(cache / f"{sample['sequence_id']}.json") for sample in samples}
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    protocol = {"schema_version": "1.0", "created_at": datetime.now(UTC).isoformat(), "seed": SEED,
                "manifest_sha256": manifest_sha, "manifest": str(manifest_path.resolve()),
                "pose_cache": str(cache.resolve()), "pose_cache_identity": identity,
                "pose_cache_identity_sha256": sha256(cache / "cache-identity.json"),
                "development_cache_sha256": fingerprints, "source_sha256": sources,
                "class_names": CLASS_NAMES, "feature_names": list(FEATURE_NAMES),
                "descriptor_settings": {"min_conf": MIN_CONF},
                "split_ids": {split: [s["sequence_id"] for s in manifest["samples"] if s["split"] == split]
                              for split in ("train", "val", "test")},
                "candidates": [0, 16], "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.01},
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch": "full valid real train plus one raw zero",
                "loss": "class-balanced CE(valid real train) + 0.25 * CE(single synthetic raw zero, class 0)",
                "normalization": {"fit": "valid real train only", "std": "population", "scale_floor": SCALE_FLOOR},
                "epoch_selection": "minimum unweighted valid real validation CE; no synthetic controls in CE",
                "threshold_selection": "real val BA/F1; static_zero_score < threshold; then closest 0.5/higher",
                "candidate_selection": "eligible only: real val BA/F1, lower loss, fewer head parameters",
                "unknown_policy": "explicit abstention, excluded metrics with classwise coverage; never impute zero",
                "synthetic_controls": "val first/last repeated; distinct zero descriptor; not real negative cases",
                "reused_test_split": True, "test_policy": "separate phase, frozen winner, one attempt; not a new holdout",
                "limits": ["no subject IDs; not subject-independent", "sparse full-sequence motion; not real-time validated",
                           "no campus validation", "static synthetic constraint does not prove dynamic negative generalization"],
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "device": "cpu"}}
    write_json(output / "protocol.json", protocol)
    records, controls, _ = read_cohort(samples, cache, manifest_sha, identity, fingerprints)
    train = [row for row in records if row["split"] == "train"]
    val = [row for row in records if row["split"] == "val"]
    write_json(output / "development-descriptors.json", {"feature_names": list(FEATURE_NAMES), "records": records})
    write_json(output / "data-quality.json", {"train": coverage(train), "val": coverage(val),
                                             "unknown_sequences": [r for r in records if not r["valid"]]})
    for name, rows in (("train", train), ("val", val)):
        if {r["label"] for r in rows if r["valid"]} != {0, 1}:
            raise ValueError(f"both classes must have valid real {name} descriptors; quality report saved")
    features, _ = valid_tensors(train)
    mean, scale = features.mean(0), features.std(0, unbiased=False).clamp_min(SCALE_FLOOR)
    candidates = [train_candidate(hidden, train, val, controls, mean, scale, output) for hidden in (0, 16)]
    eligible = [(result, model) for result, model in candidates if result["threshold_selection"]["eligible"]]
    if not eligible:
        write_json(output / "selection-failed.json", {"reason": "no candidate passes synthetic static gate"})
        raise ValueError("no eligible candidate; do not consult test data")
    def rank(pair):
        metrics = pair[0]["validation"]["metrics"]
        return metrics["balanced_accuracy"], metrics["f1"], -metrics["loss"], -pair[0]["parameter_count"]
    winner, model = max(eligible, key=rank)
    threshold = winner["threshold_selection"]["threshold"]
    descriptor_sha = sources["src/campus_safety_ai/training/pose_motion.py"]
    checkpoint = {"schema_version": "1.0", "model_name": "pose-motion-classifier", "model_state": model.state_dict(),
                  "feature_dim": len(FEATURE_NAMES), "hidden_dim": winner["hidden_dim"], "feature_names": list(FEATURE_NAMES),
                  "class_names": CLASS_NAMES, "threshold": threshold, "manifest_sha256": manifest_sha,
                  "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"],
                  "protocol_sha256": sha256(output / "protocol.json"), "descriptor_source_sha256": descriptor_sha,
                  "descriptor_settings": {"min_conf": MIN_CONF}, "best_epoch": winner["best_epoch"]}
    torch.save(checkpoint, output / "best.pt")
    checkpoint_sha = sha256(output / "best.pt")
    validation = {**winner["validation"], "split": "val", "checkpoint_sha256": checkpoint_sha,
                  "manifest_sha256": manifest_sha, "threshold_selection": winner["threshold_selection"]}
    write_json(output / "validation.json", validation)
    write_json(output / "static-gate.json", winner["static_gate"])
    export_head(model, val, output, checkpoint_sha, threshold)
    artifacts = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    frozen = {"schema_version": "1.0", "frozen_at": datetime.now(UTC).isoformat(),
              "checkpoint_sha256": checkpoint_sha, "validation_sha256": sha256(output / "validation.json"),
              "protocol_sha256": sha256(output / "protocol.json"), "export_sha256": sha256(output / "export.json"),
              "manifest_sha256": manifest_sha, "descriptor_source_sha256": descriptor_sha,
              "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"], "source_sha256": sources,
              "artifact_sha256": artifacts, "hidden_dim": winner["hidden_dim"], "threshold": threshold,
              "best_epoch": winner["best_epoch"], "reused_test_split": True}
    write_json(output / "frozen.json", frozen)
    print(json.dumps({"phase": "train", "hidden_dim": winner["hidden_dim"], "threshold": threshold,
                      "validation": validation["metrics"], "coverage": validation["coverage"],
                      "static_false_positives": winner["static_gate"]["false_positive_count"]}), flush=True)
    return frozen


def verify_frozen(manifest_path: Path, cache: Path, output: Path) -> tuple[dict, dict, dict, dict]:
    manifest, manifest_sha = read_manifest(manifest_path)
    frozen = json.loads((output / "frozen.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    if frozen["manifest_sha256"] != manifest_sha or protocol["manifest_sha256"] != manifest_sha:
        raise ValueError("frozen manifest SHA mismatch")
    for name, expected in frozen["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source changed after freeze: {name}")
    for name, expected in frozen["artifact_sha256"].items():
        if Path(name).name != name or sha256(output / name) != expected:
            raise ValueError(f"artifact changed after freeze: {name}")
    for field, name in (("checkpoint_sha256", "best.pt"), ("validation_sha256", "validation.json"),
                        ("protocol_sha256", "protocol.json"), ("export_sha256", "export.json")):
        if frozen[field] != sha256(output / name):
            raise ValueError(f"frozen {field} mismatch")
    identity = read_identity(cache, manifest_sha)
    if identity != protocol["pose_cache_identity"] or sha256(cache / "cache-identity.json") != protocol["pose_cache_identity_sha256"]:
        raise ValueError("pose cache identity changed after freeze")
    for name, expected in protocol["development_cache_sha256"].items():
        if sha256(cache / f"{name}.json") != expected:
            raise ValueError(f"development cache changed after freeze: {name}")
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint["threshold"] != frozen["threshold"] or checkpoint["hidden_dim"] != frozen["hidden_dim"]
            or checkpoint["manifest_sha256"] != manifest_sha or checkpoint["protocol_sha256"] != frozen["protocol_sha256"]
            or checkpoint["pose_checkpoint_sha256"] != frozen["pose_checkpoint_sha256"]
            or checkpoint["descriptor_source_sha256"] != frozen["descriptor_source_sha256"]
            or checkpoint["feature_names"] != list(FEATURE_NAMES) or checkpoint["class_names"] != CLASS_NAMES):
        raise ValueError("frozen checkpoint metadata mismatch")
    validation = json.loads((output / "validation.json").read_text())
    gate = json.loads((output / "static-gate.json").read_text())
    if (validation["threshold"] != frozen["threshold"] or validation["split"] != "val"
            or validation["checkpoint_sha256"] != frozen["checkpoint_sha256"]
            or not gate["eligible"] or gate["canonical_raw_zero_score"] >= frozen["threshold"]):
        raise ValueError("frozen validation/static gate mismatch")
    return manifest, identity, frozen, checkpoint


def test_phase(manifest_path: Path, cache: Path, output: Path) -> dict:
    if (output / "test.json").exists() or (output / "test-attempt.json").exists():
        raise ValueError("test has already been attempted; refusing repeat or overwrite")
    manifest, identity, frozen, checkpoint = verify_frozen(manifest_path, cache, output)
    samples = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    if not all((cache / f"{sample['sequence_id']}.json").is_file() for sample in samples):
        raise ValueError("all 14 test caches must exist before the single evaluation attempt")
    write_json(output / "test-attempt.json", {"started_at": datetime.now(UTC).isoformat(),
                                             "frozen_sha256": sha256(output / "frozen.json"),
                                             "policy": "one attempt; do not rerun test after failure"})
    try:
        records, _, fingerprints = read_cohort(samples, cache, frozen["manifest_sha256"], identity)
        model = PoseMotionClassifier(checkpoint["feature_dim"], checkpoint["hidden_dim"]).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        result = {**evaluate(model, records, frozen["threshold"]), "split": "test", "reused_test_split": True,
                  "interpretation": "reused 14-sequence development comparison, not a new independent holdout",
                  "class_names": CLASS_NAMES, "checkpoint_sha256": frozen["checkpoint_sha256"],
                  "manifest_sha256": frozen["manifest_sha256"], "frozen_sha256": sha256(output / "frozen.json"),
                  "pose_cache_sha256": fingerprints,
                  "unknown_sequences": [row for row in records if not row["valid"]]}
        write_json(output / "test.json", result)
    except Exception as error:
        write_json(output / "test-failed.json", {"error_type": type(error).__name__, "message": str(error),
                                                "test_attempt_consumed": True})
        raise
    print(json.dumps({"phase": "test", "metrics": result["metrics"], "coverage": result["coverage"],
                      "confusion_matrix": result["confusion_matrix"], "reused_test_split": True}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/urfd-sequences-v1.json")
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    (train_phase if args.phase == "train" else test_phase)(args.manifest, args.pose_cache, args.output)


if __name__ == "__main__":
    main()
