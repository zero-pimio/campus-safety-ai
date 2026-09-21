from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def _limit(samples: list[FightSample], maximum: int | None, seed: int = 20260826) -> list[FightSample]:
    if maximum is not None and maximum <= 0:
        raise ValueError("sample limits must be positive")
    if maximum is None or len(samples) <= maximum:
        return list(samples)
    strata: dict[tuple[str, int], list[FightSample]] = defaultdict(list)
    for sample in samples:
        strata[(sample.dataset, sample.label)].append(sample)
    if maximum < len(strata):
        raise ValueError("sample limit must include every dataset/label stratum")
    generator = random.Random(seed)
    pools = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda sample: sample.video_path)
        generator.shuffle(pool)
        pools.append(pool)
    selected: list[FightSample] = []
    while len(selected) < maximum:
        for pool in pools:
            if pool and len(selected) < maximum:
                selected.append(pool.pop())
    generator.shuffle(selected)
    return selected


def _validate_manifest(samples: list[FightSample], project_root: Path) -> None:
    paths: dict[Path, str] = {}
    groups: dict[str, str] = {}
    for sample in samples:
        if sample.split not in {"train", "val", "test"} or sample.label not in {0, 1}:
            raise ValueError("manifest needs train/val/test splits and binary labels")
        if not sample.dataset.strip() or not sample.group_id.strip():
            raise ValueError("manifest dataset and group_id must be non-empty")
        path = (project_root / sample.video_path).resolve()
        if path in paths:
            if paths[path] != sample.split:
                raise ValueError(f"video path leaks across splits: {sample.video_path}")
            raise ValueError(f"duplicate video path in manifest: {sample.video_path}")
        paths[path] = sample.split
        if sample.group_id in groups and groups[sample.group_id] != sample.split:
            raise ValueError(f"group_id leaks across splits: {sample.group_id}")
        groups[sample.group_id] = sample.split
        if not path.is_file():
            raise FileNotFoundError(f"manifest video does not exist: {path}")


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "frames", "image_size"):
        value = getattr(arguments, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    accumulation_steps = getattr(arguments, "accumulation_steps", 1)
    if not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    if not isinstance(getattr(arguments, "freeze_batch_norm", False), bool):
        raise ValueError("freeze_batch_norm must be a boolean")
    for name in ("workers", "seed", "patience"):
        value = getattr(arguments, name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("max_train_samples", "max_val_samples"):
        value = getattr(arguments, name, None)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'learning_rate' else 'non-negative'}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_summary(samples: list[FightSample]) -> dict[str, Any]:
    counts = Counter((sample.dataset, sample.label) for sample in samples)
    return {
        "count": len(samples),
        "group_count": len({sample.group_id for sample in samples}),
        "by_dataset_label": [
            {"dataset": dataset, "label": label, "count": count}
            for (dataset, label), count in sorted(counts.items())
        ],
    }


def _validate_checkpoint(checkpoint: dict, frame_count: int, image_size: int) -> None:
    expected = {"frame_count": frame_count, "image_size": image_size, "class_names": ["non_fight", "fight"]}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"initialization checkpoint {key} does not match this experiment")
    if checkpoint.get("model_name") != "fight-tsn-mobilenet-v3-small" or "model_state" not in checkpoint:
        raise ValueError("initialization checkpoint must contain a compatible fight TSN model")


def _write_run(path: Path, run: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    optimizer: Any | None,
    *,
    accumulation_steps: int = 1,
    freeze_batch_norm: bool = False,
) -> Metrics:
    import torch

    training = optimizer is not None
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train(training)
    if training and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    loss_sum = 0.0
    loss_denominator = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    group_denominator = 0.0
    group_batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def step_group() -> None:
        # Gradients accumulate weighted loss sums, then receive one normalization
        # for the complete group, including any short final group.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(group_denominator)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            logits = model(frames)
            loss = criterion(logits, targets)
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise ValueError("non-finite training or validation loss")
            weight = getattr(criterion, "weight", None)
            denominator = (float(weight[targets].sum().detach().cpu()) if weight is not None
                           else int(targets.numel()))
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("loss normalization denominator must be finite and positive")
            if training:
                (loss * denominator).backward()
                group_denominator += denominator
                group_batches += 1
                if group_batches == accumulation_steps:
                    step_group()
                    group_denominator = 0.0
                    group_batches = 0
            loss_sum += loss_value * denominator
            loss_denominator += denominator
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if training and group_batches:
            step_group()
    return _metrics(loss_sum / loss_denominator if loss_denominator else 0.0, labels, predictions)


def train(arguments: argparse.Namespace) -> Path:
    # Reject destructive or invalid requests before importing torch or downloading weights.
    _validate_arguments(arguments)
    project_root = arguments.project_root.resolve()
    manifest_path = (project_root / arguments.manifest).resolve()
    output_argument = getattr(arguments, "output_dir", None)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output_dir = ((project_root / output_argument).resolve() if output_argument is not None
                  else project_root / "runtime/training" / f"fight-tsn-{run_id}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"output directory must be empty; existing experiment is preserved: {output_dir}")
    samples = read_manifest(manifest_path)
    _validate_manifest(samples, project_root)
    train_samples = _limit([sample for sample in samples if sample.split == "train"],
                           arguments.max_train_samples, arguments.seed)
    val_samples = _limit([sample for sample in samples if sample.split == "val"],
                         arguments.max_val_samples, arguments.seed)
    if not train_samples or not val_samples:
        raise ValueError("manifest must contain non-empty train and val splits")
    for name, selected in (("train", train_samples), ("val", val_samples)):
        if {sample.label for sample in selected} != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")
    init_argument = getattr(arguments, "init_checkpoint", None)
    init_path = (project_root / init_argument).resolve() if init_argument is not None else None
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {init_path}")
    manifest_sha256 = _sha256(manifest_path)
    init_sha256 = _sha256(init_path) if init_path is not None else None
    hyperparameters = {
        name: getattr(arguments, name) for name in (
            "epochs", "batch_size", "learning_rate", "weight_decay", "frames", "image_size",
            "workers", "device", "seed", "no_pretrained", "max_train_samples", "max_val_samples",
        )
    }
    hyperparameters["patience"] = getattr(arguments, "patience", 0)
    hyperparameters["accumulation_steps"] = getattr(arguments, "accumulation_steps", 1)
    hyperparameters["freeze_batch_norm"] = getattr(arguments, "freeze_batch_norm", False)
    hyperparameters["effective_batch_size"] = arguments.batch_size * hyperparameters["accumulation_steps"]
    hyperparameters["scheduler"] = {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 1}
    run: dict[str, Any] = {
        "schema_version": "1.0", "run_id": run_id, "status": "initializing",
        "started_at": datetime.now(UTC).isoformat(), "output_dir": str(output_dir),
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
        "initialization": {"kind": "fine_tune" if init_path else "new_model",
                           "checkpoint": str(init_path) if init_path else None,
                           "checkpoint_sha256": init_sha256},
        "hyperparameters": hyperparameters,
        "samples": {"train": _sample_summary(train_samples), "val": _sample_summary(val_samples)},
        "selected_samples": {"train": [sample.video_path for sample in train_samples],
                             "val": [sample.video_path for sample in val_samples]},
        "device": None, "torch_version": None, "epochs_completed": 0, "best_epoch": None,
        "improved_epochs": [], "selection_rule": "higher_val_f1_then_lower_val_loss",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    # Exclusive creation prevents two trainers from claiming the same empty directory.
    with run_path.open("x", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    best_path = output_dir / "best.pt"
    started = time.monotonic()
    try:
        import torch
        from torch.utils.data import DataLoader

        random.seed(arguments.seed)
        torch.manual_seed(arguments.seed)
        device = _select_device(arguments.device)
        run.update(device=device, torch_version=str(torch.__version__))
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True) if init_path else None
        if checkpoint is not None:
            _validate_checkpoint(checkpoint, arguments.frames, arguments.image_size)
        model = FightTsn(arguments.frames, pretrained=checkpoint is None and not arguments.no_pretrained)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        train_dataset = FightVideoDataset(project_root, train_samples, arguments.frames, arguments.image_size, True)
        val_dataset = FightVideoDataset(project_root, val_samples, arguments.frames, arguments.image_size, False)
        train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True,
                                  num_workers=arguments.workers)
        val_loader = DataLoader(val_dataset, batch_size=arguments.batch_size, shuffle=False,
                                num_workers=arguments.workers)
        counts = [sum(sample.label == label for sample in train_samples) for label in (0, 1)]
        weights = torch.tensor([len(train_samples) / (2 * count) for count in counts],
                               dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        val_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate,
                                      weight_decay=arguments.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, threshold=0, threshold_mode="abs",
        )
        provenance = {key: run[key] for key in ("run_id", "manifest_sha256", "initialization",
                                               "hyperparameters", "samples")}

        def save_best(epoch: int, metrics: Metrics) -> None:
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save({
                "model_state": model.state_dict(), "model_name": "fight-tsn-mobilenet-v3-small",
                "frame_count": arguments.frames, "image_size": arguments.image_size,
                "class_names": ["non_fight", "fight"], "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256, "seed": arguments.seed, "best_epoch": epoch,
                "val_metrics": asdict(metrics), "provenance": provenance,
            }, temporary)
            temporary.replace(best_path)
            run.update(best_epoch=epoch, best_validation_metrics=asdict(metrics))

        best_f1 = -1.0
        best_loss = math.inf
        if checkpoint is not None:
            baseline = _run_epoch(model, val_loader, val_criterion, device, None)
            best_f1 = baseline.f1
            best_loss = baseline.loss
            run["initial_validation_metrics"] = asdict(baseline)
            save_best(0, baseline)
            scheduler.step(best_f1)
        run["status"] = "running"
        _write_run(run_path, run)
        print(json.dumps({"run_id": run_id, "output_dir": str(output_dir), "device": device,
                          "initial_validation_metrics": run.get("initial_validation_metrics")}, ensure_ascii=False))
        stale_epochs = 0
        with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
            for epoch in range(1, arguments.epochs + 1):
                epoch_started = time.monotonic()
                learning_rate = float(optimizer.param_groups[0]["lr"])
                train_metrics = _run_epoch(
                    model, train_loader, criterion, device, optimizer,
                    accumulation_steps=hyperparameters["accumulation_steps"],
                    freeze_batch_norm=hyperparameters["freeze_batch_norm"],
                )
                val_metrics = _run_epoch(model, val_loader, val_criterion, device, None)
                scheduler.step(val_metrics.f1)
                record = {"epoch": epoch, "device": device, "learning_rate": learning_rate,
                          "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                          "effective_batch_size": hyperparameters["effective_batch_size"],
                          "epoch_seconds": time.monotonic() - epoch_started,
                          "train": asdict(train_metrics), "val": asdict(val_metrics)}
                improved = val_metrics.f1 > best_f1 or (
                    val_metrics.f1 == best_f1 and val_metrics.loss < best_loss
                )
                record["improved"] = improved
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, ensure_ascii=False))
                if improved:
                    best_f1 = val_metrics.f1
                    best_loss = val_metrics.loss
                    stale_epochs = 0
                    save_best(epoch, val_metrics)
                    run["improved_epochs"].append(epoch)
                else:
                    stale_epochs += 1
                run["epochs_completed"] = epoch
                _write_run(run_path, run)
                if hyperparameters["patience"] and stale_epochs >= hyperparameters["patience"]:
                    run["status"] = "early_stopped"
                    break
        if run["status"] == "running":
            run["status"] = "completed"
    except BaseException as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        run["error_type"] = type(error).__name__
        raise
    finally:
        run["finished_at"] = datetime.now(UTC).isoformat()
        run["elapsed_seconds"] = time.monotonic() - started
        _write_run(run_path, run)
    print(f"best checkpoint -> {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight fight video classifier")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/fight-v1.csv"))
    parser.add_argument("--output-dir", type=Path, help="empty directory; defaults to a unique experiment directory")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="trusted local checkpoint for a new fine-tuning experiment; optimizer state is reset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="microbatches per optimizer update; the last group uses its actual loss denominator")
    parser.add_argument("--freeze-batch-norm", action="store_true",
                        help="keep BatchNorm running statistics fixed while learning its affine parameters")
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
    parser.add_argument("--patience", type=int, default=0, help="stop after this many epochs without better val F1; 0 disables")
    arguments = parser.parse_args()
    train(arguments)


if __name__ == "__main__":
    main()
