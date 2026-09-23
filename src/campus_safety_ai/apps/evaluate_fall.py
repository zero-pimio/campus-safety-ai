from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campus_safety_ai.core.event_metrics import evaluate_fall_events


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            rows.append(value)
    return rows


def evaluate(
    ground_truth_path: Path,
    events_path: Path,
    observations_path: Path,
    output_path: Path,
    *,
    tolerance_seconds: float = 0,
    max_observation_gap_seconds: float = 2,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output_path}")
    truth = json.loads(ground_truth_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    report = evaluate_fall_events(
        truth, _read_jsonl(events_path), _read_jsonl(observations_path),
        tolerance_seconds=tolerance_seconds, max_observation_gap_seconds=max_observation_gap_seconds,
    )
    report["inputs"] = {
        "ground_truth": str(ground_truth_path.resolve()),
        "events": str(events_path.resolve()),
        "observations": str(observations_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate continuous fall alarms and observation coverage")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True, help="local EventRecord JSONL with sourceEpoch")
    parser.add_argument("--observations", type=Path, required=True, help="fall observation JSONL; null score is unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0)
    parser.add_argument("--max-observation-gap-seconds", type=float, default=2)
    args = parser.parse_args()
    try:
        report = evaluate(
            args.ground_truth, args.events, args.observations, args.output,
            tolerance_seconds=args.tolerance_seconds,
            max_observation_gap_seconds=args.max_observation_gap_seconds,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"metrics": report["metrics"], "warnings": report["warnings"]}, ensure_ascii=False))
    print(f"evaluation -> {args.output}")


if __name__ == "__main__":
    main()
