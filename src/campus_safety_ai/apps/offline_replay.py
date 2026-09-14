from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.apps.common import default_analysis
from campus_safety_ai.contracts import Detections
from campus_safety_ai.core.event_analysis import EventAnalysis
from campus_safety_ai.core.event_delivery import Destination, EventDelivery, JsonlDestination
from campus_safety_ai.settings import PROJECT_ROOT, load_platform_settings


def run(
    input_path: Path,
    output_path: Path,
    analysis: EventAnalysis | None = None,
    *,
    destination: Destination | None = None,
    outbox_path: Path | None = None,
) -> int:
    analysis = analysis or default_analysis()
    delivery = EventDelivery(
        outbox_path or output_path.with_suffix(".sqlite3"),
        destination or JsonlDestination(output_path),
    )
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
    parser.add_argument(
        "--event-config",
        type=Path,
        default=PROJECT_ROOT / "configs/events/intrusion-v1.toml",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/gate-02.toml",
    )
    parser.add_argument("--tracker", choices=("simple_iou", "bytetrack"))
    parser.add_argument("--platform-config", type=Path)
    arguments = parser.parse_args()
    destination = None
    outbox_path = None
    if arguments.platform_config:
        platform = load_platform_settings(arguments.platform_config)
        destination = build_platform_destination(platform, events_path=arguments.output)
        outbox_path = platform.outbox
    count = run(
        arguments.input,
        arguments.output,
        default_analysis(arguments.event_config, arguments.scene_config, arguments.tracker),
        destination=destination,
        outbox_path=outbox_path,
    )
    print(f"produced {count} event records -> {arguments.output}")


if __name__ == "__main__":
    main()
