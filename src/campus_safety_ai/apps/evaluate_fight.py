from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    samples = [sample for sample in read_manifest(manifest_path) if sample.split == split]
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"manifest contains no samples for split={split}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame_count = int(checkpoint["frame_count"])
    image_size = int(checkpoint["image_size"])
    selected_device = _select_device(device)
    model = FightTsn(frame_count=frame_count, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(selected_device)
    dataset = FightVideoDataset(project_root.resolve(), samples, frame_count, image_size, False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    criterion = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    fight_scores: list[float] = []
    with torch.inference_mode():
        for frames, targets in loader:
            frames = frames.to(selected_device)
            targets = targets.to(selected_device)
            logits = model(frames)
            losses.append(float(criterion(logits, targets).detach().cpu()))
            labels.extend(int(value) for value in targets.detach().cpu().tolist())
            predictions.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().tolist())
            fight_scores.extend(float(value) for value in logits.softmax(dim=1)[:, 1].detach().cpu().tolist())

    metrics = classification_metrics(sum(losses) / max(1, len(losses)), labels, predictions)
    misclassified = [
        {
            "video_path": sample.video_path,
            "expected": label,
            "predicted": prediction,
            "fight_score": score,
            "dataset": sample.dataset,
            "group_id": sample.group_id,
        }
        for sample, label, prediction, score in zip(samples, labels, predictions, fight_scores)
        if label != prediction
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model_name": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_validation_metrics": checkpoint.get("val_metrics"),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": split,
        "sample_count": len(samples),
        "label_counts": {str(key): value for key, value in sorted(Counter(labels).items())},
        "device": selected_device,
        "metrics": asdict(metrics),
        "confusion_matrix": confusion_matrix(labels, predictions),
        "misclassified": misclassified,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    arguments = parser.parse_args()
    if not arguments.manifest.is_absolute():
        arguments.manifest = arguments.project_root / arguments.manifest
    if not arguments.checkpoint.is_absolute():
        arguments.checkpoint = arguments.project_root / arguments.checkpoint
    if not arguments.output.is_absolute():
        arguments.output = arguments.project_root / arguments.output
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
    )


if __name__ == "__main__":
    main()
