from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Callable
from urllib.parse import urlparse

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.adapters.evidence import DirectoryEvidenceSink, NullEvidenceSink
from campus_safety_ai.adapters.runtimes.fight_factory import build_fight_classifier
from campus_safety_ai.adapters.runtimes.paddle_fight import PaddlePpTsmFightClassifier
from campus_safety_ai.adapters.video_sources import OpenCvVideoSource, redact_video_source
from campus_safety_ai.apps.fight_replay import default_fight_analysis
from campus_safety_ai.contracts import iso_time, parse_time
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import Destination, EventDelivery, JsonlDestination
from campus_safety_ai.core.fight_analysis import FightEventAnalysis
from campus_safety_ai.core.fight_inference import FightClassifier
from campus_safety_ai.core.fight_pipeline import EvidenceSink, FightPipelineResult, FightVideoPipeline
from campus_safety_ai.settings import (
    PROJECT_ROOT,
    load_fight_model_settings,
    load_fight_policy,
    load_platform_settings,
    load_video_runtime_settings,
)


def _execute(
    video_path: str | Path,
    classifier: FightClassifier,
    camera_id: str,
    started_at: datetime,
    source_epoch: int = 1,
    frame_len: int | None = None,
    sample_frequency: int = 7,
    analysis: FightEventAnalysis | None = None,
    evidence: EvidenceSink | None = None,
    consume: Callable[[FightPipelineResult], None] | None = None,
) -> FightPipelineResult:
    requested_frame_len = frame_len or classifier.frame_len
    if requested_frame_len != classifier.frame_len:
        raise ValueError(
            f"runtime frame_count={requested_frame_len} does not match "
            f"classifier frame_len={classifier.frame_len}"
        )
    source = OpenCvVideoSource(video_path, requested_frame_len, sample_frequency)
    pipeline = FightVideoPipeline(
        classifier,
        analysis or default_fight_analysis(),
        evidence or NullEvidenceSink(),
    )
    with source:
        if consume is None:
            return pipeline.run(source, camera_id, source_epoch, started_at)
        for batch in pipeline.stream(source, camera_id, source_epoch, started_at):
            consume(batch)
        return FightPipelineResult((), (), source.fps, source.sample_frequency)


def analyze_video(
    video_path: str | Path,
    classifier: FightClassifier,
    camera_id: str,
    started_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
    result = _execute(video_path, classifier, camera_id, started_at)
    return (
        list(result.observations),
        [record.to_dict() for record in result.events],
        result.fps,
        result.sample_frequency,
    )


def _source_stem(source: str | Path, camera_id: str) -> str:
    value = str(source)
    if "://" in value:
        return Path(urlparse(value).path).stem or camera_id
    return Path(value).stem


def run(
    video_path: str | Path,
    model_dir: Path,
    output_dir: Path,
    camera_id: str,
    started_at: datetime,
    *,
    classifier: FightClassifier | None = None,
    analysis: FightEventAnalysis | None = None,
    evidence: EvidenceSink | None = None,
    source_epoch: int = 1,
    frame_len: int = 8,
    sample_frequency: int = 7,
    events_path: Path | None = None,
    outbox_path: Path | None = None,
    destination: Destination | None = None,
    collect_results: bool = True,
    background_delivery: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classifier = classifier or PaddlePpTsmFightClassifier(model_dir)
    stem = _source_stem(video_path, camera_id)
    observations_path = output_dir / f"{stem}-observations.jsonl"
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    events_path = events_path or output_dir / f"{stem}-events.jsonl"
    outbox_path = outbox_path or events_path.with_suffix(".sqlite3")
    observations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    window_count = 0
    event_count = 0
    delivery_type = BackgroundDelivery if background_delivery else EventDelivery
    with delivery_type(
        outbox_path,
        destination or JsonlDestination(events_path),
    ) as delivery, observations_path.open("w", encoding="utf-8") as handle:
        def consume(batch: FightPipelineResult) -> None:
            nonlocal window_count, event_count
            delivery.submit(list(batch.events))
            for item in batch.observations:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                start, end = item["sampledFrameRange"]
                print(f"window={item['sequence']} frames={start}-{end} fight_score={item['score']:.6f}")
            handle.flush()
            window_count += len(batch.observations)
            event_count += len(batch.events)
            if collect_results:
                observations.extend(batch.observations)
                events.extend(record.to_dict() for record in batch.events)

        result = _execute(
            video_path, classifier, camera_id, started_at, source_epoch,
            frame_len, sample_frequency, analysis, evidence, consume,
        )

    print(
        f"video={redact_video_source(video_path)} fps={result.fps:.3f} "
        f"sample_frequency={result.sample_frequency} "
        f"windows={window_count} events={event_count}"
    )
    print(f"observations -> {observations_path}")
    print(f"events -> {events_path}")
    print(f"outbox -> {outbox_path}")
    return observations, events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify fight windows from an MP4 or RTSP source and emit project contracts"
    )
    parser.add_argument("--video", required=True, help="local video path or RTSP URL")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtimes/pc-dev.toml",
    )
    parser.add_argument("--platform-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--model-dir", type=Path, help="legacy override for a Paddle PP-TSM directory"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/fight-video"))
    parser.add_argument("--camera-id", default="offline-video-01")
    parser.add_argument("--source-epoch", type=int, default=1)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--outbox", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument(
        "--started-at",
        help="timezone-aware ISO timestamp; defaults to the current UTC time",
    )
    arguments = parser.parse_args()
    started_at = (
        parse_time(arguments.started_at)
        if arguments.started_at
        else datetime.now(UTC)
    )
    runtime = load_video_runtime_settings(arguments.runtime_config)
    model_config = arguments.model_config or runtime.fight_model_config
    classifier = (
        PaddlePpTsmFightClassifier(arguments.model_dir)
        if arguments.model_dir
        else build_fight_classifier(load_fight_model_settings(model_config))
    )
    policy = load_fight_policy(runtime.fight_event_config)
    platform = load_platform_settings(arguments.platform_config or runtime.platform_config)
    evidence_dir = arguments.evidence_dir or runtime.evidence_dir
    evidence = (
        NullEvidenceSink()
        if arguments.no_evidence or not runtime.evidence_enabled
        else DirectoryEvidenceSink(evidence_dir, policy.start_score)
    )
    print(f"video_started_at={iso_time(started_at)}")
    events_path = arguments.events or platform.events
    outbox_path = arguments.outbox or platform.outbox
    destination = build_platform_destination(platform, events_path=events_path)
    run(
        arguments.video,
        arguments.model_dir or Path("models/ppTSM"),
        arguments.output_dir,
        arguments.camera_id,
        started_at,
        classifier=classifier,
        analysis=FightEventAnalysis(policy),
        evidence=evidence,
        source_epoch=arguments.source_epoch,
        frame_len=runtime.frame_count,
        sample_frequency=runtime.sample_frequency,
        events_path=events_path,
        outbox_path=outbox_path,
        destination=destination,
        collect_results=False,
        background_delivery=platform.destination == "easyaiot",
    )


if __name__ == "__main__":
    main()
