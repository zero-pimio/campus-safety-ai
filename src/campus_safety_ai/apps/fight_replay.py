from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.contracts import BehaviorObservation
from campus_safety_ai.core.event_delivery import EventDelivery, JsonlDestination
from campus_safety_ai.core.fight_analysis import FightEventAnalysis, FightPolicy


def default_fight_analysis() -> FightEventAnalysis:
    return FightEventAnalysis(
        FightPolicy(
            edge_id="edge-dev-01",
            start_score=0.75,
            end_score=0.35,
            confirm_seconds=2.0,
            clear_seconds=2.0,
            cooldown_seconds=10.0,
            config_version="gate-02-fight-v1",
        )
    )


def run(input_path: Path, output_path: Path) -> int:
    analysis = default_fight_analysis()
    delivery = EventDelivery(output_path.with_suffix(".sqlite3"), JsonlDestination(output_path))
    produced = 0
    last_observation: BehaviorObservation | None = None
    with delivery:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = BehaviorObservation.from_dict(json.loads(line))
                last_observation = observation
                records = analysis.advance(observation)
                produced += len(records)
                delivery.submit(records)
        if last_observation is not None:
            # The replay is a finished source epoch; close anything still open.
            records = analysis.finalize(
                last_observation.camera_id,
                last_observation.source_epoch,
                last_observation.observed_at,
            )
            produced += len(records)
            delivery.submit(records)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay fight scores through event analysis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    count = run(arguments.input, arguments.output)
    print(f"produced {count} fight event records -> {arguments.output}")


if __name__ == "__main__":
    main()

