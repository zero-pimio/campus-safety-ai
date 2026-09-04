from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from campus_safety_ai.training.manifest import FightSample, read_manifest
from campus_safety_ai.training.metrics import Metrics, classification_metrics
from campus_safety_ai.training.model import FightTsn
from campus_safety_ai.training.video_dataset import FightVideoDataset


def _metrics(loss: float, labels: list[int], predictions: list[int]) -> Metrics:
    return classification_metrics(loss, labels, predictions)


def _select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _limit(samples: list[FightSample], maximum: int | None) -> list[FightSample]:
    if maximum is None or len(samples) <= maximum:
        return samples
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    selected: list[FightSample] = []
    for label in (0, 1):
        selected.extend(by_label[label][: max(1, maximum // 2)])
    return selected[:maximum]


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    optimizer: Any | None,
) -> Metrics:
    import torch

    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(targets.numel())
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return _metrics(loss_sum / max(1, len(labels)), labels, predictions)


def train(arguments: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    project_root = arguments.project_root.resolve()
    samples = read_manifest(arguments.manifest)
    train_samples = _limit([sample for sample in samples if sample.split == "train"], arguments.max_train_samples)
    val_samples = _limit([sample for sample in samples if sample.split == "val"], arguments.max_val_samples)
    if not train_samples or not val_samples:
        raise ValueError("manifest must contain non-empty train and val splits")

    train_dataset = FightVideoDataset(project_root, train_samples, arguments.frames, arguments.image_size, True)
    val_dataset = FightVideoDataset(project_root, val_samples, arguments.frames, arguments.image_size, False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
    )
    device = _select_device(arguments.device)
    model = FightTsn(arguments.frames, pretrained=not arguments.no_pretrained).to(device)
    counts = [sum(sample.label == label for sample in train_samples) for label in (0, 1)]
    weights = torch.tensor(
        [len(train_samples) / (2 * max(1, count)) for count in counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.output_dir / "metrics.jsonl"
    best_path = arguments.output_dir / "best.pt"
    best_f1 = -1.0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, arguments.epochs + 1):
            train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = _run_epoch(model, val_loader, criterion, device, None)
            record = {
                "epoch": epoch,
                "device": device,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()
            print(json.dumps(record, ensure_ascii=False))
            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": "fight-tsn-mobilenet-v3-small",
                        "frame_count": arguments.frames,
                        "image_size": arguments.image_size,
                        "class_names": ["non_fight", "fight"],
                        "manifest": str(arguments.manifest),
                        "seed": arguments.seed,
                        "val_metrics": asdict(val_metrics),
                    },
                    best_path,
                )
    print(f"best checkpoint -> {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight fight video classifier")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/fight-v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/training/fight-tsn-v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    arguments = parser.parse_args()
    if not arguments.manifest.is_absolute():
        arguments.manifest = arguments.project_root / arguments.manifest
    if not arguments.output_dir.is_absolute():
        arguments.output_dir = arguments.project_root / arguments.output_dir
    train(arguments)


if __name__ == "__main__":
    main()
