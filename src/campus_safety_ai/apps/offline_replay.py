from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.apps.common import default_analysis
from campus_safety_ai.contracts import Detections
from campus_safety_ai.core.event_delivery import EventDelivery, JsonlDestination


def run(input_path: Path, output_path: Path) -> int:
    analysis = default_analysis()
    delivery = EventDelivery(output_path.with_suffix(".sqlite3"), JsonlDestination(output_path))
    produced = 0
    try:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records = analysis.advance(Detections.from_dict(json.loads(line)))
                produced += len(records)
                delivery.submit(records)
    finally:
        delivery.close()
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay fixed detections through event analysis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    count = run(arguments.input, arguments.output)
    print(f"produced {count} event records -> {arguments.output}")


if __name__ == "__main__":
    main()

