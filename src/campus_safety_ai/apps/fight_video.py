from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

from campus_safety_ai.adapters.runtimes.paddle_fight import (
    FightPrediction,
    PaddlePpTsmFightClassifier,
)
from campus_safety_ai.apps.fight_replay import default_fight_analysis
from campus_safety_ai.contracts import BehaviorObservation, iso_time, parse_time


@dataclass(frozen=True)
class VideoWindow:
    frames: tuple[Any, ...]
    first_frame: int
    last_frame: int


class FightClassifier(Protocol):
    model_version: str

    def predict(self, rgb_frames: Sequence[Any]) -> FightPrediction: ...


def read_video_windows(
    video_path: Path, frame_len: int = 8, sample_frequency: int = 7
) -> tuple[float, int, Iterator[VideoWindow]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed; run scripts/setup_video_runtime.sh"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count < frame_len:
        capture.release()
        raise ValueError(
            f"video must report a positive FPS and contain at least {frame_len} frames"
        )

    effective_frequency = min(sample_frequency, max(1, frame_count // frame_len))

    def windows() -> Iterator[VideoWindow]:
        sampled: list[Any] = []
        sampled_indices: list[int] = []
        frame_index = 0
        try:
            while True:
                available, bgr_frame = capture.read()
                if not available:
                    break
                if frame_index % effective_frequency == 0:
                    sampled.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                    sampled_indices.append(frame_index)
                    if len(sampled) == frame_len:
                        yield VideoWindow(
                            frames=tuple(sampled),
                            first_frame=sampled_indices[0],
                            last_frame=sampled_indices[-1],
                        )
                        sampled.clear()
                        sampled_indices.clear()
                frame_index += 1
        finally:
            capture.release()

    return fps, effective_frequency, windows()


def analyze_video(
    video_path: Path,
    classifier: FightClassifier,
    camera_id: str,
    started_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
    fps, sample_frequency, windows = read_video_windows(video_path)
    analysis = default_fight_analysis()
    observations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for sequence, window in enumerate(windows, start=1):
        prediction = classifier.predict(window.frames)
        window_started_at = started_at + timedelta(seconds=window.first_frame / fps)
        window_ended_at = started_at + timedelta(seconds=window.last_frame / fps)
        observation = BehaviorObservation(
            camera_id=camera_id,
            source_epoch=1,
            sequence=sequence,
            observed_at=window_ended_at,
            window_started_at=window_started_at,
            window_ended_at=window_ended_at,
            behavior="fighting",
            score=prediction.fight_score,
            model_version=classifier.model_version,
        )
        value = observation.to_dict()
        value["sampledFrameRange"] = [window.first_frame, window.last_frame]
        value["logits"] = list(prediction.logits)
        observations.append(value)
        events.extend(record.to_dict() for record in analysis.advance(observation))

    return observations, events, fps, sample_frequency


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def run(
    video_path: Path,
    model_dir: Path,
    output_dir: Path,
    camera_id: str,
    started_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classifier = PaddlePpTsmFightClassifier(model_dir)
    observations, events, fps, sample_frequency = analyze_video(
        video_path, classifier, camera_id, started_at
    )
    observations_path = output_dir / f"{video_path.stem}-observations.jsonl"
    events_path = output_dir / f"{video_path.stem}-events.jsonl"
    _write_jsonl(observations_path, observations)
    _write_jsonl(events_path, events)

    for item in observations:
        start, end = item["sampledFrameRange"]
        print(
            f"window={item['sequence']} frames={start}-{end} "
            f"fight_score={item['score']:.6f}"
        )
    print(
        f"video={video_path} fps={fps:.3f} sample_frequency={sample_frequency} "
        f"windows={len(observations)} events={len(events)}"
    )
    print(f"observations -> {observations_path}")
    print(f"events -> {events_path}")
    return observations, events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify fight windows in a local video and emit project contracts"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/ppTSM"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/fight-video"))
    parser.add_argument("--camera-id", default="offline-video-01")
    parser.add_argument(
        "--started-at",
        help="timezone-aware ISO timestamp; defaults to the current UTC time",
    )
    arguments = parser.parse_args()
    started_at = (
        parse_time(arguments.started_at)
        if arguments.started_at
        else datetime.now(timezone.utc)
    )
    print(f"video_started_at={iso_time(started_at)}")
    run(
        arguments.video,
        arguments.model_dir,
        arguments.output_dir,
        arguments.camera_id,
        started_at,
    )


if __name__ == "__main__":
    main()
