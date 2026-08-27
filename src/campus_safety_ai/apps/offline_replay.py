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
    last_batch: Detections | None = None
    with delivery:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                batch = Detections.from_dict(json.loads(line))
                last_batch = batch
                records = analysis.advance(batch)
                produced += len(records)
                delivery.submit(records)
        if last_batch is not None:
            # The replay is a finished source epoch; close anything still open.
            records = analysis.finalize(
                last_batch.camera_id, last_batch.source_epoch, last_batch.captured_at
            )
            produced += len(records)
            delivery.submit(records)
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

