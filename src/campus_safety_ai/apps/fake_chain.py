from __future__ import annotations

from pathlib import Path

from campus_safety_ai.apps.offline_replay import run


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "tests/golden/intrusion/detections.jsonl"
    destination = root / "runtime/demo-events.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.with_suffix(".sqlite3").unlink(missing_ok=True)
    count = run(source, destination)
    print(f"G1 fake chain passed: {count} records written to {destination}")


if __name__ == "__main__":
    main()

