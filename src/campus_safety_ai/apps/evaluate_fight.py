from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from campus_safety_ai.training.calibration import (
    predictions_at_threshold,
    select_validation_threshold,
    threshold_from_validation_report,
    validate_threshold,
)
from campus_safety_ai.training.manifest import read_manifest
from campus_safety_ai.training.metrics import classification_metrics, confusion_matrix
from campus_safety_ai.training.model import FightTsn
from campus_safety_ai.training.video_dataset import FightVideoDataset


def _select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_arguments(
    split: str,
    output_path: Path,
    batch_size: int,
    workers: int,
    max_samples: int | None,
    threshold: float | None,
    select_threshold: bool,
    threshold_report: Path | None,
) -> None:
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    for name, value, minimum in (("batch_size", batch_size, 1), ("workers", workers, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer")
    if sum((threshold is not None, select_threshold, threshold_report is not None)) > 1:
        raise ValueError("threshold, select_threshold, and threshold_report are mutually exclusive")
    if threshold is not None:
        validate_threshold(threshold)
    if select_threshold and split != "val":
        raise ValueError("threshold selection requires the val split; never fit on test data")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")


def _summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [record["label"] for record in records]
    predictions = [record["predicted"] for record in records]
    return {
        "sample_count": len(records),
        "label_counts": {str(key): value for key, value in sorted(Counter(labels).items())},
        "metrics": asdict(classification_metrics(
            sum(record["loss"] for record in records) / len(records), labels, predictions
        )),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }


def evaluate(
    project_root: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    output_path: Path,
    batch_size: int = 8,
    workers: int = 0,
    device: str = "auto",
    max_samples: int | None = None,
    *,
    threshold: float | None = None,
    select_threshold: bool = False,
    threshold_report: Path | None = None,
) -> dict[str, Any]:
    _validate_arguments(
        split, output_path, batch_size, workers, max_samples, threshold, select_threshold, threshold_report
    )
    started = time.perf_counter()
    samples = [sample for sample in read_manifest(manifest_path) if sample.split == split]
    available_sample_count = len(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"manifest contains no samples for split={split}")
    partial = len(samples) < available_sample_count
    if select_threshold and partial:
        raise ValueError("threshold selection requires the complete validation split")
    if select_threshold and {sample.label for sample in samples} != {0, 1}:
        raise ValueError("threshold selection requires both binary labels")
    checkpoint_sha256 = _sha256(checkpoint_path)
    manifest_sha256 = _sha256(manifest_path)
    decision_threshold = 0.5 if threshold is None else validate_threshold(threshold)
    threshold_source: dict[str, Any] = {"kind": "default" if threshold is None else "explicit"}
    if threshold_report is not None:
        source_report = json.loads(threshold_report.read_text(encoding="utf-8"))
        decision_threshold = threshold_from_validation_report(
            source_report, checkpoint_sha256=checkpoint_sha256, manifest_sha256=manifest_sha256
        )
        threshold_source = {
            "kind": "validation_report",
            "path": str(threshold_report.resolve()),
            "sha256": _sha256(threshold_report),
            "split": "val",
        }

    import torch
    from torch.utils.data import DataLoader

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame_count = int(checkpoint["frame_count"])
    image_size = int(checkpoint["image_size"])
    selected_device = _select_device(device)
    model = FightTsn(frame_count=frame_count, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(selected_device)
    dataset = FightVideoDataset(project_root.resolve(), samples, frame_count, image_size, False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []
    labels: list[int] = []
    fight_scores: list[float] = []
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for frames, targets in loader:
            frames = frames.to(selected_device)
            targets = targets.to(selected_device)
            logits = model(frames)
            losses.extend(float(value) for value in criterion(logits, targets).detach().cpu().tolist())
            labels.extend(int(value) for value in targets.detach().cpu().tolist())
            fight_scores.extend(float(value) for value in logits.softmax(dim=1)[:, 1].detach().cpu().tolist())
    evaluation_seconds = time.perf_counter() - evaluation_started
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("model produced a non-finite evaluation loss")
    selection = None
    if select_threshold:
        selection = select_validation_threshold(labels, fight_scores, split=split, partial=partial)
        decision_threshold = selection["threshold"]
        threshold_source = {"kind": "validation_selection", "split": "val"}
    predictions = predictions_at_threshold(fight_scores, decision_threshold)

    records = [
        {
            "video_path": sample.video_path,
            "label": label,
            "expected": label,
            "predicted": prediction,
            "fight_score": score,
            "dataset": sample.dataset,
            "group_id": sample.group_id,
            "loss": loss,
        }
        for sample, label, prediction, score, loss in zip(
            samples, labels, predictions, fight_scores, losses, strict=True
        )
    ]
    report: dict[str, Any] = {
        "schema_version": "1.1",
        "model_name": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_validation_metrics": checkpoint.get("val_metrics"),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "split": split,
        "available_sample_count": available_sample_count,
        "partial": partial,
        "max_samples": max_samples,
        "device": selected_device,
        "frame_count": frame_count,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "threshold": decision_threshold,
        "decision_rule": "fight_score >= threshold",
        "threshold_source": threshold_source,
        "threshold_selection": selection,
        **_summarize_predictions(records),
        "predictions": records,
        "by_dataset": {
            name: _summarize_predictions([record for record in records if record["dataset"] == name])
            for name in sorted({sample.dataset for sample in samples})
        },
        "misclassified": [record for record in records if record["label"] != record["predicted"]],
        "timing": {
            "evaluation_seconds": evaluation_seconds,
            "samples_per_second": len(samples) / evaluation_seconds if evaluation_seconds else None,
            "mean_seconds_per_sample": evaluation_seconds / len(samples),
            "total_seconds": time.perf_counter() - started,
            "scope": "evaluation loop includes video decode, preprocessing, transfers, model, and loss; "
                     "not pure model latency; total also includes setup and threshold selection, excludes report write",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(json.dumps({key: report[key] for key in ("split", "sample_count", "metrics", "confusion_matrix")}, ensure_ascii=False))
    print(f"evaluation -> {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained fight classifier on a fixed manifest split")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/fight-v1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--threshold", type=float, help="fixed fight-score threshold (default: 0.5)")
    thresholds.add_argument(
        "--select-threshold", action="store_true", help="fit balanced-accuracy threshold on complete val split only"
    )
    thresholds.add_argument(
        "--threshold-report", type=Path,
        help="freeze threshold from a complete val report with matching checkpoint and manifest SHA256",
    )
    arguments = parser.parse_args()
    if not arguments.manifest.is_absolute():
        arguments.manifest = arguments.project_root / arguments.manifest
    if not arguments.checkpoint.is_absolute():
        arguments.checkpoint = arguments.project_root / arguments.checkpoint
    if not arguments.output.is_absolute():
        arguments.output = arguments.project_root / arguments.output
    if arguments.threshold_report is not None and not arguments.threshold_report.is_absolute():
        arguments.threshold_report = arguments.project_root / arguments.threshold_report
    evaluate(
        arguments.project_root,
        arguments.manifest,
        arguments.checkpoint,
        arguments.split,
        arguments.output,
        arguments.batch_size,
        arguments.workers,
        arguments.device,
        arguments.max_samples,
        threshold=arguments.threshold,
        select_threshold=arguments.select_threshold,
        threshold_report=arguments.threshold_report,
    )


if __name__ == "__main__":
    main()
