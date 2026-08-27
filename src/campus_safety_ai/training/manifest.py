from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path


FIGHT_ACTIONS = frozenset({"fight", "punch", "kick", "push", "slap", "choke"})
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class FightSample:
    video_path: str
    label: int
    dataset: str
    group_id: str
    split: str = ""
    action_labels: tuple[str, ...] = ()
    license_status: str = "research-only-unverified"

    def to_row(self) -> dict[str, str | int]:
        return {
            "video_path": self.video_path,
            "label": self.label,
            "dataset": self.dataset,
            "group_id": self.group_id,
            "split": self.split,
            "action_labels": ",".join(self.action_labels),
            "license_status": self.license_status,
        }


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _read_airtlab_actions(path: Path) -> dict[str, tuple[str, ...]]:
    actions: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            actions[row[0].strip()] = tuple(
                item.strip().lower() for item in row[1].split(",") if item.strip()
            )
    return actions


def _collect_scfd(project_root: Path) -> list[FightSample]:
    root = project_root / "datasets/private/scfd"
    samples: list[FightSample] = []
    for directory, label in (("fight", 1), ("noFight", 0)):
        for path in sorted((root / directory).glob("*.mp4")):
            samples.append(
                FightSample(
                    video_path=_relative(path, project_root),
                    label=label,
                    dataset="scfd",
                    group_id=f"scfd:{path.stem}",
                    action_labels=("fight",) if label else ("non-fight",),
                    license_status="research-only-third-party-video-rights-unverified",
                )
            )
    return samples


def _collect_airtlab(project_root: Path) -> list[FightSample]:
    root = project_root / "datasets/private/airtlab/violence-detection-dataset"
    violent_actions = _read_airtlab_actions(root / "violent-action-classes.csv")
    nonviolent_actions = _read_airtlab_actions(root / "nonviolent-action-classes.csv")
    samples: list[FightSample] = []

    for directory, label, action_map in (
        ("non-violent", 0, nonviolent_actions),
        ("violent", 1, violent_actions),
    ):
        for camera in ("cam1", "cam2"):
            for path in sorted((root / directory / camera).glob("*.mp4")):
                actions = action_map.get(path.name, ())
                if label == 1 and FIGHT_ACTIONS.isdisjoint(actions):
                    continue
                samples.append(
                    FightSample(
                        video_path=_relative(path, project_root),
                        label=label,
                        dataset="airtlab",
                        group_id=f"airtlab:{directory}:{path.stem}",
                        action_labels=actions,
                        license_status="research-and-education",
                    )
                )
    return samples


def _assign_splits(samples: list[FightSample], seed: int) -> list[FightSample]:
    assignments: dict[str, str] = {}
    strata = sorted({(sample.dataset, sample.label) for sample in samples})
    for dataset, label in strata:
        groups = sorted(
            {sample.group_id for sample in samples if sample.dataset == dataset and sample.label == label},
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
        )
        if len(groups) < 3:
            raise ValueError(f"{dataset} label={label} needs at least three independent groups")
        train_end = max(1, int(len(groups) * 0.70))
        val_end = max(train_end + 1, train_end + int(len(groups) * 0.15))
        val_end = min(val_end, len(groups) - 1)
        for index, group in enumerate(groups):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            assignments[group] = split
    return [replace(sample, split=assignments[sample.group_id]) for sample in samples]


def build_fight_samples(project_root: Path, seed: int = 20260826) -> list[FightSample]:
    samples = [*_collect_scfd(project_root), *_collect_airtlab(project_root)]
    if not samples:
        raise FileNotFoundError("no SCFD or AIRTLab videos found under datasets/private")
    return _assign_splits(samples, seed)


def write_manifest(samples: list[FightSample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].to_row()))
        writer.writeheader()
        writer.writerows(sample.to_row() for sample in samples)


def read_manifest(path: Path) -> list[FightSample]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            FightSample(
                video_path=row["video_path"],
                label=int(row["label"]),
                dataset=row["dataset"],
                group_id=row["group_id"],
                split=row["split"],
                action_labels=tuple(filter(None, row["action_labels"].split(","))),
                license_status=row["license_status"],
            )
            for row in csv.DictReader(handle)
        ]
