"""Train and export a small, explicitly offline URFD RGB sequence classifier.

The protocol is written before features are extracted. Test frames are first
processed after the selected checkpoint and validation threshold are frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from campus_safety_ai.training.calibration import predictions_at_threshold, select_validation_threshold
from campus_safety_ai.training.fall_model import FallTemporalClassifier
from campus_safety_ai.training.metrics import classification_metrics, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260921
PREPROCESS = {
    "color": "RGB", "frame_count": 16, "image_size": 224,
    "resize": "PIL bilinear, preserve aspect ratio, center letterbox",
    "padding_rgb": [124, 116, 104], "scale": "divide by 255",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "sampling": "16 equal-bin centers over RGB frames with official sync timestamps; not an online window",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(manifest: dict, root: Path) -> list[dict]:
    if (manifest.get("dataset") != "urfd-cam0" or manifest.get("seed") != SEED
            or manifest.get("frame_count") != 16 or manifest.get("class_names") != ["non_fall", "fall"]):
        raise ValueError("unexpected URFD manifest protocol")
    samples = manifest["samples"]
    expected = {("train", 0): 24, ("train", 1): 18, ("val", 0): 8,
                ("val", 1): 6, ("test", 0): 8, ("test", 1): 6}
    if Counter((sample["split"], sample["label"]) for sample in samples) != expected:
        raise ValueError("expected all 70 sequences and the frozen 42/14/14 split")
    ids, groups, paths, hashes = set(), set(), set(), {}
    for sample in samples:
        sequence = sample["sequence_id"]
        if sequence in ids or sample["group_id"] in groups:
            raise ValueError("duplicate sequence or group across the manifest")
        ids.add(sequence)
        groups.add(sample["group_id"])
        expected_label = 1 if sequence.startswith("fall-") else 0 if sequence.startswith("adl-") else -1
        if sample["label"] != expected_label or sample["group_id"] != f"urfd:{sequence}":
            raise ValueError("sequence label/group does not match the original category")
        if any(len(sample[key]) != 16 for key in ("frames", "frame_numbers", "frame_timestamps_ms", "members")):
            raise ValueError("every sequence must have 16 aligned frames")
        for values in (sample["frame_numbers"], sample["frame_timestamps_ms"]):
            if not all(math.isfinite(v) for v in values) or any(a >= b for a, b in zip(values, values[1:], strict=False)):
                raise ValueError("frame numbers and timestamps must strictly increase")
        for relative, member in zip(sample["frames"], sample["members"], strict=True):
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or path in paths or not path.is_file():
                raise ValueError("invalid, missing or duplicate frame path")
            paths.add(path)
            digest = sha256(path)
            if digest != member["sha256"]:
                raise ValueError(f"frame checksum mismatch: {path}")
            if digest in hashes and hashes[digest] != sample["split"]:
                raise ValueError("identical frame bytes cross data splits")
            hashes[digest] = sample["split"]
    # Verify exact prespecified membership, not just split counts.
    for prefix, count, train, val in (("fall", 30, 18, 6), ("adl", 40, 24, 8)):
        ordered = [f"{prefix}-{i:02d}" for i in range(1, count + 1)]
        random.Random(SEED).shuffle(ordered)
        planned = {name: "train" if i < train else "val" if i < train + val else "test"
                   for i, name in enumerate(ordered)}
        for sample in samples:
            if sample["sequence_id"].startswith(prefix) and planned.get(sample["sequence_id"]) != sample["split"]:
                raise ValueError("split membership differs from the preregistered plan")
    return samples


def load_frames(sample: dict, root: Path = ROOT) -> torch.Tensor:
    frames = []
    mean = np.array(PREPROCESS["mean"], dtype=np.float32)
    std = np.array(PREPROCESS["std"], dtype=np.float32)
    for relative in sample["frames"]:
        with Image.open(root / relative) as source:
            source = source.convert("RGB")
            width, height = source.size
            ratio = 224 / max(width, height)
            size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
            resized = source.resize(size, Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (224, 224), tuple(PREPROCESS["padding_rgb"]))
            canvas.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
            array = (np.asarray(canvas, dtype=np.float32) / 255 - mean) / std
            frames.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
    return torch.stack(frames).unsqueeze(0)


def extract_features(model, samples: list[dict], device: str) -> torch.Tensor:
    model.to(device).eval()
    outputs = []
    tick = time.monotonic()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            features = model.forward_frame_features(load_frames(sample).to(device)).cpu()
            if not torch.isfinite(features).all():
                raise ValueError("nonfinite frame features")
            outputs.append(features)
            print(f"features {index + 1}/{len(samples)} {sample['sequence_id']} "
                  f"{sample['split']} elapsed={time.monotonic() - tick:.1f}s", flush=True)
    model.cpu()
    # Clone outside inference_mode so these constants can train the linear head.
    return torch.cat(outputs).clone()


def fit_normalization(model, train_features: torch.Tensor) -> None:
    with torch.no_grad():
        summary = model.summarize_features(train_features)
        model.set_feature_normalization(summary.mean(0), summary.std(0, unbiased=False).clamp_min(0.05))


def evaluate(model, features, samples, threshold=0.5) -> dict:
    labels = [sample["label"] for sample in samples]
    with torch.inference_mode():
        logits = model.forward_features(features)
        loss = float(torch.nn.functional.cross_entropy(logits, torch.tensor(labels)))
        scores = logits.softmax(1)[:, 1].tolist()
    predictions = predictions_at_threshold(scores, threshold)
    return {"threshold": threshold, "sample_count": len(samples),
            "metrics": asdict(classification_metrics(loss, labels, predictions)),
            "confusion_matrix": confusion_matrix(labels, predictions),
            "predictions": [{"sequence_id": sample["sequence_id"], "label": label,
                             "fall_score": score, "prediction": prediction}
                            for sample, label, score, prediction in zip(samples, labels, scores, predictions, strict=True)]}


def train_candidate(model, initial_head, train_features, train_samples, val_features, val_samples,
                    weight_decay: float, output: Path) -> dict:
    torch.manual_seed(SEED)
    model.head.load_state_dict(copy.deepcopy(initial_head))
    labels = torch.tensor([sample["label"] for sample in train_samples])
    counts = torch.bincount(labels, minlength=2)
    weights = len(labels) / (2 * counts.float())
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.003, weight_decay=weight_decay)
    best_loss, best_epoch, best_head = math.inf, 0, None
    history = []
    # Full-batch optimization is cheap on 42 frozen features and deterministic.
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.forward_features(train_features), labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        validation = evaluate(model, val_features, val_samples)
        row = {"epoch": epoch, "train_weighted_loss": float(loss.detach()), "val": validation["metrics"]}
        history.append(row)
        if validation["metrics"]["loss"] < best_loss - 1e-7:
            best_loss, best_epoch = validation["metrics"]["loss"], epoch
            best_head = copy.deepcopy(model.head.state_dict())
        if epoch % 10 == 0 or epoch == 1:
            print(f"train wd={weight_decay} epoch={epoch} train_loss={float(loss.detach()):.5f} "
                  f"val_loss={validation['metrics']['loss']:.5f} val_f1={validation['metrics']['f1']:.3f}", flush=True)
        if epoch - best_epoch >= 30:
            break
    model.head.load_state_dict(best_head)
    parameter_delta = math.sqrt(sum(float((best_head[key] - initial_head[key]).square().sum())
                                    for key in initial_head))
    if parameter_delta <= 0:
        raise ValueError("training did not change the classifier parameters")
    candidate = {"weight_decay": weight_decay, "learning_rate": 0.003, "best_epoch": best_epoch,
                 "head_parameter_delta_l2": parameter_delta,
                 "epochs_completed": len(history), "history": history,
                 "validation_at_0_5": evaluate(model, val_features, val_samples)}
    write_json(output / f"candidate-wd-{weight_decay}.json", candidate)
    torch.save(best_head, output / f"candidate-wd-{weight_decay}-head.pt")
    return candidate


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/urfd-sequences-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing to overwrite a nonempty experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    samples = validate_manifest(json.loads(args.manifest.read_text()), ROOT)
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    test_samples = [s for s in samples if s["split"] == "test"]
    manifest_sha = sha256(args.manifest)
    model = FallTemporalClassifier(pretrained=True).eval()
    protocol = {
        "created_at": datetime.now(UTC).isoformat(), "seed": SEED,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
        "source_sha256": {relative: sha256(ROOT / relative) for relative in (
            "scripts/train_urfd_baseline.py", "scripts/prepare_urfd.py",
            "src/campus_safety_ai/training/fall_model.py",
            "src/campus_safety_ai/training/calibration.py",
            "src/campus_safety_ai/training/metrics.py")},
        "task": "offline whole-sequence fall/non-fall classification",
        "class_names": ["non_fall", "fall"], "preprocess": PREPROCESS,
        "data_quality_exclusions": [{"sequence_id": s["sequence_id"], "split": s["split"],
                                     "excluded_unsynchronized_frame_numbers": s["excluded_unsynchronized_frame_numbers"]}
                                    for s in samples if s.get("excluded_unsynchronized_frame_numbers")],
        "backbone": "ImageNet MobileNetV3-Small IMAGENET1K_V1; frozen, eval mode",
        "backbone_checkpoint_sha256": sha256(Path(torch.hub.get_dir()) / "checkpoints/mobilenet_v3_small-047dcff4.pth"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "summary": "mean,max,last-minus-first -> 1728 features",
        "normalization": "mean/population_std fitted on train only, scale floor 0.05",
        "candidates": [{"weight_decay": wd, "learning_rate": 0.003} for wd in [0.01, 0.1]],
        "optimizer": "AdamW, full batch, class-balanced cross entropy, max 200 epochs",
        "epoch_selection": "minimum unweighted validation CE, patience 30, min_delta 1e-7",
        "candidate_selection": "validation at 0.5: balanced_accuracy, f1, lower loss, lower weight_decay",
        "threshold_selection": "winner only, validation BA then F1 then nearest 0.5 then higher",
        "test_policy": "test inference once after checkpoint and validation threshold are frozen",
        "split_ids": {split: [s["sequence_id"] for s in samples if s["split"] == split]
                      for split in ("train", "val", "test")},
        "demo_ids_chosen_before_training": [next(s["sequence_id"] for s in val_samples if s["label"] == label)
                                            for label in (0, 1)],
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__), "feature_device": args.device,
                        "head_device": "cpu", "platform": platform.platform()},
        "limits": ["sequence split, no subject IDs: not subject-independent",
                   "full-sequence future frames: not a realtime fall detector",
                   "14 test sequences only; no campus/clinical validation",
                   "dataset CC BY-NC-SA 4.0; academic noncommercial experiment"],
    }
    write_json(args.output / "protocol.json", protocol)
    features = extract_features(model, train_samples + val_samples, args.device)
    torch.save({"sequence_ids": [s["sequence_id"] for s in train_samples + val_samples],
                "manifest_sha256": manifest_sha, "features": features}, args.output / "train-val-features.pt")
    train_features, val_features = features[:len(train_samples)], features[len(train_samples):]
    fit_normalization(model, train_features)
    initial_head = copy.deepcopy(model.head.state_dict())
    torch.save(initial_head, args.output / "initial-head.pt")
    write_json(args.output / "initial-untrained-validation.json", evaluate(model, val_features, val_samples))
    candidates = [train_candidate(model, initial_head, train_features, train_samples, val_features, val_samples,
                                  wd, args.output) for wd in (0.01, 0.1)]
    def rank(candidate):
        metrics = candidate["validation_at_0_5"]["metrics"]
        return metrics["balanced_accuracy"], metrics["f1"], -metrics["loss"], -candidate["weight_decay"]
    winner = max(candidates, key=rank)
    model.head.load_state_dict(torch.load(args.output / f"candidate-wd-{winner['weight_decay']}-head.pt",
                                          map_location="cpu", weights_only=True))
    model.eval()
    initial_validation = evaluate(model, val_features, val_samples)
    calibration = select_validation_threshold([s["label"] for s in val_samples],
                                             [p["fall_score"] for p in initial_validation["predictions"]], split="val")
    checkpoint_path = args.output / "best.pt"
    torch.save({"model_name": "fall-temporal-mobilenet-v3-small", "model_state": model.state_dict(),
                "class_names": ["non_fall", "fall"], "frame_count": 16, "image_size": 224,
                "preprocess": PREPROCESS, "manifest_sha256": manifest_sha,
                "threshold": calibration["threshold"], "selected_weight_decay": winner["weight_decay"],
                "best_epoch": winner["best_epoch"], "protocol_sha256": sha256(args.output / "protocol.json")},
               checkpoint_path)
    checkpoint_sha = sha256(checkpoint_path)
    validation = {**evaluate(model, val_features, val_samples, calibration["threshold"]),
                  "split": "val", "partial": False, "available_sample_count": len(val_samples),
                  "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
                  "threshold_selection": calibration}
    write_json(args.output / "validation.json", validation)
    write_json(args.output / "selection-frozen.json", {
        "frozen_at": datetime.now(UTC).isoformat(), "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": sha256(args.output / "protocol.json"), "manifest_sha256": manifest_sha,
        "validation_sha256": sha256(args.output / "validation.json"),
        "threshold": calibration["threshold"], "selected_weight_decay": winner["weight_decay"],
        "best_epoch": winner["best_epoch"]})
    # Reload the delivered checkpoint, then validate export using held-out validation inputs only.
    loaded = FallTemporalClassifier(pretrained=False).eval()
    loaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state"])
    example = load_frames(val_samples[0])
    with torch.inference_mode():
        reloaded_logits = loaded(example)
        cached_logits = model.forward_features(val_features[:1])
    reload_error = float((reloaded_logits - cached_logits).abs().max())
    if reload_error > 1e-3:
        raise ValueError(f"reloaded RGB model differs from feature model: {reload_error}")
    onnx_path = args.output / "model.onnx"
    torch.onnx.export(loaded, example, onnx_path, input_names=["frames"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    parity = []
    for sample in [next(s for s in val_samples if s["label"] == label) for label in (0, 1)]:
        frames = load_frames(sample)
        with torch.inference_mode():
            expected = loaded(frames).numpy()
        tick = time.monotonic()
        actual = session.run(None, {"frames": frames.numpy()})[0]
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-3:
            raise ValueError(f"ONNX parity failed: {error}")
        parity.append({"sequence_id": sample["sequence_id"], "maximum_logit_error": error,
                       "onnx_cpu_seconds": time.monotonic() - tick})
    write_json(args.output / "export.json", {"onnx_sha256": sha256(onnx_path),
               "checkpoint_sha256": checkpoint_sha, "input_shape": [1, 16, 3, 224, 224],
               "output": "2 logits in order non_fall,fall; softmax then frozen threshold",
               "preprocess": PREPROCESS, "threshold": calibration["threshold"],
               "reloaded_model_max_logit_error": reload_error, "validation_parity": parity,
               "elapsed_seconds": time.monotonic() - started})
    print("Selection frozen; now running the untouched test split once.", flush=True)
    test_features = extract_features(model, test_samples, args.device)
    test = {**evaluate(model, test_features, test_samples, calibration["threshold"]), "split": "test",
            "partial": False, "available_sample_count": len(test_samples),
            "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
            "selection_frozen_sha256": sha256(args.output / "selection-frozen.json")}
    matrix = test["confusion_matrix"]
    test["wilson_95_percent_intervals"] = {
        "accuracy": wilson(matrix["true_positive"] + matrix["true_negative"], len(test_samples)),
        "recall": wilson(matrix["true_positive"], 6), "specificity": wilson(matrix["true_negative"], 8)}
    majority = [0] * len(test_samples)
    test["always_non_fall_baseline"] = asdict(classification_metrics(0, [s["label"] for s in test_samples], majority))
    test["always_non_fall_baseline"].pop("loss")
    write_json(args.output / "test.json", test)
    print(json.dumps({"complete": str(args.output.resolve()), "test": test["metrics"],
                      "confusion_matrix": matrix}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
