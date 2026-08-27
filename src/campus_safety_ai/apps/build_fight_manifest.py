from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from campus_safety_ai.training.manifest import build_fight_samples, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe fight training manifest")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("datasets/manifests/fight-v1.csv"))
    parser.add_argument("--seed", type=int, default=20260826)
    arguments = parser.parse_args()
    samples = build_fight_samples(arguments.project_root, arguments.seed)
    output = arguments.output
    if not output.is_absolute():
        output = arguments.project_root / output
    write_manifest(samples, output)
    counts = Counter((sample.split, sample.label) for sample in samples)
    print(f"wrote {len(samples)} videos -> {output}")
    for split in ("train", "val", "test"):
        print(f"{split}: non_fight={counts[split, 0]} fight={counts[split, 1]}")


if __name__ == "__main__":
    main()
