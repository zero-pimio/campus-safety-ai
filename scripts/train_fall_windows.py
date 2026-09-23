#!/usr/bin/env python3
"""Train/freeze a causal fixed-window candidate, then evaluate a held-out subject once."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from campus_safety_ai.adapters.runtimes.pose_fall import FrozenPoseFallHead, sha256
from campus_safety_ai.core.fall_analysis import FallPolicy
from campus_safety_ai.core.fall_pipeline import WindowPolicy
from campus_safety_ai.training import pose_motion
from campus_safety_ai.training.fall_windows import collect_windows, event_report, validate_manifest
from campus_safety_ai.training.metrics import classification_metrics, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260923
CLASS_NAMES = ["no_fall_transition", "fall_transition"]
THRESHOLDS = [.35, .5, .65, .8, .9]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def policy(threshold):
    return FallPolicy(edge_id="fixed-window-evaluation", start_score=threshold, end_score=min(.25, threshold - .1),
                      confirm_seconds=.2, clear_seconds=1, cooldown_seconds=2, config_version="fall-window-web-v1")


def tensors(rows):
    valid = [r for r in rows if r["features"] is not None and r["label"] is not None]
    if not valid or set(r["label"] for r in valid) != {0, 1}:
        raise ValueError("both observed classes required; do not relabel unknown/uncertain windows")
    return valid, torch.tensor([r["features"] for r in valid]), torch.tensor([r["label"] for r in valid])


def scores_for(model, rows):
    indices = [i for i, r in enumerate(rows) if r["features"] is not None]
    scores = [None] * len(rows)
    if indices:
        with torch.inference_mode():
            values = model(torch.tensor([rows[i]["features"] for i in indices])).softmax(1)[:, 1].tolist()
        for i, value in zip(indices, values, strict=True):
            scores[i] = value
    return scores


def metrics(rows, scores, threshold):
    selected = [(r["label"], s) for r, s in zip(rows, scores, strict=True)
                if r["label"] is not None and s is not None]
    labels = [pair[0] for pair in selected]
    predictions = [int(pair[1] >= threshold) for pair in selected]
    if not selected:
        return {"loss": None, "accuracy": None, "balanced_accuracy": None, "precision": None, "recall": None,
                "f1": None, "confusion": confusion_matrix([], []), "labeled_known_windows": 0,
                "total_observations": len(rows), "unknown_observations": sum(s is None for s in scores)}
    loss = float(np.mean([-np.log(max(1e-12, s if label else 1 - s)) for label, s in selected]))
    return {**asdict(classification_metrics(loss, labels, predictions)), "confusion": confusion_matrix(labels, predictions),
            "labeled_known_windows": len(labels), "total_observations": len(rows),
            "unknown_observations": sum(s is None for s in scores)}


def compact_report(report):
    return {key: {"metrics": report[key]["metrics"], "coverage": report[key]["coverage"]}
            for key in ("strict", "tolerance_0_5s")}


def train(args, manifest, window):
    args.output.mkdir(parents=True, exist_ok=False)
    samples = [s for s in manifest["samples"] if s["split"] in {"train", "val"}]
    rows = [row for sample in samples for row in collect_windows(sample, args.cache, window)]
    train_rows, val_rows = ([r for r in rows if r["split"] == split] for split in ("train", "val"))
    valid_train, x, y = tensors(train_rows)
    valid_val, vx, vy = tensors(val_rows)
    manifest_sha = sha256(args.manifest)
    identity = json.loads((args.cache / "identity.json").read_text())
    baseline_cache = getattr(args, "baseline_cache", None) or args.cache
    baseline_identity = json.loads((baseline_cache / "identity.json").read_text())
    baseline = FrozenPoseFallHead(args.baseline)
    if any(baseline_identity[key] != baseline.extraction_identity[key]
           for key in ("pose_checkpoint_sha256", "settings")):
        raise ValueError("baseline poses must use the baseline frozen detector and extraction settings")
    protocol = {"created_at": datetime.now(UTC).isoformat(), "seed": SEED,
                "manifest_sha256": manifest_sha, "pose_cache_identity": {**identity, "window_policy": asdict(window)},
                "cache_sha256": {s["sample_id"]: sha256(args.cache / (s["sample_id"] + ".json")) for s in samples},
                "window_policy": asdict(window), "feature_names": list(pose_motion.FEATURE_NAMES),
                "labels": "positive: >=0.2s confirmed fall motion inside causal1s window; negative: entire window verified0; other windows excluded",
                "split": "source subjects1+2 train, subject3 val, subject4 test; physical cameras unresolved",
                "normalization": "train known labeled windows only; std floor1e-6",
                "training": "two predeclared hidden_dims0,16; clip-balanced then class-balanced CE +0.25 raw-static-zero CE; AdamW lr.01 wd.01; epochs400 patience60",
                "selection": "epoch min val clip-balanced CE; candidate/threshold min strict event FN then FP then max labeled window BA; zero-motion score must be below end threshold",
                "thresholds": THRESHOLDS, "event_policy": asdict(policy(.5)),
                "test_policy": "test detector/score only after frozen.json; evaluate phase once; never tune from test",
                "baseline_sha256": sha256(args.baseline),
                "baseline_cache_identity": baseline_identity,
                "baseline_window_policy": asdict(WindowPolicy()),
                "baseline_cache_sha256": {s["sample_id"]: sha256(baseline_cache / (s["sample_id"] + ".json"))
                                          for s in samples if s["split"] == "val"},
                "source_sha256": {p: sha256(ROOT / p) for p in (
                    "scripts/train_fall_windows.py", "src/campus_safety_ai/training/fall_windows.py",
                    "src/campus_safety_ai/core/fall_pipeline.py", "src/campus_safety_ai/training/pose_motion.py")},
                "limits": ["23 short staged home videos,4 source subjects", "correlated overlapping windows are not independent videos",
                           "single-person guard abstains at occlusion/crowds", "visual boundary uncertainty", "no real recovery labels"]}
    write(args.output / "protocol.json", protocol)
    write(args.output / "development-windows.json", {"window_policy": asdict(window), "rows": rows})
    train_counts = Counter((r["sample_id"], r["label"]) for r in valid_train)
    weights = torch.tensor([1 / train_counts[(r["sample_id"], r["label"])] for r in valid_train])
    for cls in (0, 1):
        weights[y == cls] /= weights[y == cls].sum()
    weights /= weights.sum()
    val_counts = Counter(r["sample_id"] for r in valid_val)
    vweights = torch.tensor([1 / val_counts[r["sample_id"]] for r in valid_val])
    vweights /= vweights.sum()
    candidates = []
    val_samples = [s for s in samples if s["split"] == "val"]
    for hidden in (0, 16):
        torch.manual_seed(SEED)
        model = pose_motion.PoseMotionClassifier(len(pose_motion.FEATURE_NAMES), hidden)
        model.set_feature_normalization(x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        best_loss, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(1, 401):
            model.train()
            loss = (F.cross_entropy(model(x), y, reduction="none") * weights).sum()
            loss = loss + .25 * F.cross_entropy(model(torch.zeros(1, x.shape[1])), torch.zeros(1, dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_loss = float((F.cross_entropy(model(vx), vy, reduction="none") * vweights).sum())
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= 60:
                break
        model.load_state_dict(best_state)
        model.eval()
        scores = scores_for(model, val_rows)
        with torch.inference_mode():
            static_score = float(model(torch.zeros(1, x.shape[1])).softmax(1)[0, 1])
        for threshold in THRESHOLDS:
            candidate_policy = policy(threshold)
            report = event_report(val_samples, val_rows, scores, candidate_policy)
            window_metrics = metrics(val_rows, scores, threshold)
            result = {"hidden_dim": hidden, "best_epoch": best_epoch, "threshold": threshold,
                      "validation_clip_balanced_loss": best_loss, "static_score": static_score,
                      "eligible": static_score < candidate_policy.end_score,
                      "window_metrics": window_metrics, "event_metrics": compact_report(report)}
            candidates.append((result, copy.deepcopy(model.state_dict())))
    write(args.output / "candidates.json", [r for r, _ in candidates])
    eligible = [(r, state) for r, state in candidates if r["eligible"]]
    if not eligible:
        raise ValueError("no static-safe candidate; test remains unused")
    def rank(item):
        r = item[0]
        m = r["event_metrics"]["strict"]["metrics"]
        return (m["false_negatives"], m["false_positives"], -r["window_metrics"]["balanced_accuracy"],
                r["validation_clip_balanced_loss"], r["hidden_dim"], -r["threshold"])
    winner, state = min(eligible, key=rank)
    checkpoint = {"model_name": "pose-motion-classifier", "model_state": state,
                  "feature_dim": len(pose_motion.FEATURE_NAMES), "feature_names": list(pose_motion.FEATURE_NAMES),
                  "class_names": CLASS_NAMES, "hidden_dim": winner["hidden_dim"], "threshold": winner["threshold"],
                  "manifest_sha256": manifest_sha, "protocol_sha256": sha256(args.output / "protocol.json"),
                  "pose_checkpoint_sha256": identity["pose_checkpoint_sha256"],
                  "descriptor_source_sha256": sha256(Path(pose_motion.__file__)), "descriptor_settings": {"min_conf": .25}}
    torch.save(checkpoint, args.output / "best.pt")
    baseline_rows = [row for sample in val_samples for row in collect_windows(sample, baseline_cache, WindowPolicy())]
    old_scores = scores_for(baseline._model, baseline_rows)
    old_policy = FallPolicy(edge_id="fixed-window-evaluation", start_score=.5, end_score=.35,
                           confirm_seconds=.3, clear_seconds=1, cooldown_seconds=2)
    old_report = event_report(val_samples, baseline_rows, old_scores, old_policy)
    write(args.output / "validation.json", {"candidate": winner, "baseline_sha256": baseline.checkpoint_sha256,
                                          "baseline_window_metrics": metrics(baseline_rows, old_scores, .5),
                                          "baseline_events": compact_report(old_report)})
    write(args.output / "frozen.json", {"frozen_at": datetime.now(UTC).isoformat(),
                                       "checkpoint_sha256": sha256(args.output / "best.pt"),
                                       "protocol_sha256": sha256(args.output / "protocol.json"),
                                       "manifest_sha256": manifest_sha, "threshold": winner["threshold"],
                                       "event_policy": asdict(policy(winner["threshold"])),
                                       "baseline_sha256": sha256(args.baseline),
                                       "promotion": "experimental_only; default production checkpoint unchanged"})
    print(json.dumps({"output": str(args.output), "candidate": winner}, ensure_ascii=False), flush=True)


def evaluate(args, manifest, window):
    frozen = json.loads((args.output / "frozen.json").read_text())
    if frozen["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("annotations changed since model selection")
    protocol = json.loads((args.output / "protocol.json").read_text())
    baseline_cache = getattr(args, "baseline_cache", None) or args.cache
    if (asdict(window) != protocol["window_policy"]
            or {**json.loads((args.cache / "identity.json").read_text()), "window_policy": asdict(window)}
            != protocol["pose_cache_identity"] or sha256(args.baseline) != frozen["baseline_sha256"]):
        raise ValueError("evaluation settings, cache identity or baseline differ from frozen protocol")
    if ("baseline_cache_identity" in protocol
            and json.loads((baseline_cache / "identity.json").read_text()) != protocol["baseline_cache_identity"]):
        raise ValueError("baseline cache differs from frozen protocol")
    for path, expected in protocol["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError("evaluation source differs from frozen protocol")
    report_path = args.output / "test.json"
    if report_path.exists() or (args.output / "test-evaluation-started.json").exists():
        raise ValueError("test evaluation already exists; no silent retesting or tuning")
    head = FrozenPoseFallHead(args.output / "best.pt")
    baseline = FrozenPoseFallHead(args.baseline)
    samples = [s for s in manifest["samples"] if s["split"] == "test"]
    cache_hashes = {s["sample_id"]: sha256(args.cache / (s["sample_id"] + ".json")) for s in samples}
    baseline_cache_hashes = {s["sample_id"]: sha256(baseline_cache / (s["sample_id"] + ".json")) for s in samples}
    with (args.output / "test-evaluation-started.json").open("x") as handle:
        json.dump({"started_at": datetime.now(UTC).isoformat(), "frozen_sha256": sha256(args.output / "frozen.json"),
                   "cache_sha256": cache_hashes, "baseline_cache_sha256": baseline_cache_hashes,
                   "failure_policy": "do not retry automatically; audit any failure"}, handle, indent=2)
    rows = [row for sample in samples for row in collect_windows(sample, args.cache, window)]
    baseline_rows = rows if baseline_cache == args.cache and window == WindowPolicy() else [
        row for sample in samples for row in collect_windows(sample, baseline_cache, WindowPolicy())]
    write(args.output / "test-windows.json", {"rows": rows})
    result = {"frozen_sha256": sha256(args.output / "frozen.json"),
              "cache_sha256": cache_hashes, "baseline_cache_sha256": baseline_cache_hashes,
              "baseline_sha256": baseline.checkpoint_sha256,
              "independent_subject_count": len({s["subject_id"] for s in samples}), "physical_camera_independence": "unverified"}
    for name, model, event_policy, current_rows in (
        ("candidate", head._model, FallPolicy(**frozen["event_policy"]), rows),
        ("baseline", baseline._model, FallPolicy(edge_id="fixed-window-evaluation", start_score=.5, end_score=.35,
                                                confirm_seconds=.3, clear_seconds=1, cooldown_seconds=2), baseline_rows),
    ):
        scores = scores_for(model, current_rows)
        report = event_report(samples, current_rows, scores, event_policy)
        result[name] = {"window_metrics": metrics(current_rows, scores, event_policy.start_score),
                        "event_metrics": compact_report(report)}
        write(args.output / f"test-{name}-replay.json", report)
    write(report_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/fall-web-v1.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "runtime/training/fall-web-pose-v1")
    parser.add_argument("--baseline-cache", type=Path, default=ROOT / "runtime/training/fall-web-pose-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/training/fall-window-web-v1-20260923")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runtime/training/fall-pose-dense-v1-20260921/best.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    (train if args.phase == "train" else evaluate)(args, manifest, WindowPolicy())


if __name__ == "__main__":
    main()
