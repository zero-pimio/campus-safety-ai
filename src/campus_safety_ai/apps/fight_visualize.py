from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, UTC
from pathlib import Path

from campus_safety_ai.adapters.runtimes.ultralytics import UltralyticsPerception
from campus_safety_ai.contracts import FramePacket


def _load_score_timeline(path: Path) -> list[tuple[int, float]]:
    timeline: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            frame_range = value.get("sampledFrameRange")
            if not isinstance(frame_range, list) or len(frame_range) != 2:
                raise ValueError("observation is missing sampledFrameRange")
            timeline.append((int(frame_range[1]), float(value["score"])))
    if not timeline:
        raise ValueError("observation file contains no fight windows")
    return sorted(timeline)


def visualize(
    video_path: Path,
    observations_path: Path,
    output_path: Path,
    detector_path: str,
    person_confidence: float = 0.25,
    fight_threshold: float = 0.75,
) -> int:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("install the vision extra: pip install -e '.[vision]'") from error

    timeline = _load_score_timeline(observations_path)
    detector = UltralyticsPerception(
        detector_path, model_version=Path(detector_path).stem
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video metadata is invalid")

    # Keep person localization visually independent of scene classification.
    # A scene alert cannot establish which detected people are participants.
    display_scale = max(0.75, width / 960)
    header_height = 2 * round(38 * display_scale)
    footer_height = 2 * round(15 * display_scale)
    person_color = (255, 210, 40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height + header_height + footer_height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create output video: {output_path}")

    origin = datetime(1970, 1, 1, tzinfo=UTC)
    frame_index = 0
    score_index = 0
    fight_score: float | None = None
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break

            while score_index < len(timeline) and frame_index >= timeline[score_index][0]:
                fight_score = timeline[score_index][1]
                score_index += 1

            packet = FramePacket(
                camera_id="offline-visualization",
                source_epoch=1,
                sequence=frame_index,
                captured_at=origin + timedelta(seconds=frame_index / fps),
                received_at=origin + timedelta(seconds=frame_index / fps),
                width=width,
                height=height,
                image=frame,
            )
            detections = detector.detect(packet)
            people = [
                item
                for item in detections.detections
                if item.label == "person" and item.confidence >= person_confidence
            ]
            for person in people:
                box = person.bbox
                left, top, right, bottom = map(
                    int, (box.x1, box.y1, box.x2, box.y2)
                )
                cv2.rectangle(frame, (left, top), (right, bottom), person_color, 2)
                cv2.putText(
                    frame,
                    f"person {person.confidence:.2f}",
                    (left, max(18, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    person_color,
                    2,
                    cv2.LINE_AA,
                )

            fighting = fight_score is not None and fight_score >= fight_threshold
            color = (
                (30, 30, 235) if fighting
                else (60, 180, 60) if fight_score is not None
                else (180, 180, 180)
            )
            status = "FIGHT" if fighting else ("NORMAL" if fight_score is not None else "COLLECTING")
            score_text = "--" if fight_score is None else f"{fight_score:.3f}"
            if fighting:
                cv2.rectangle(frame, (2, 2), (width - 3, height - 3), color, 5)

            canvas = cv2.copyMakeBorder(
                frame, header_height, footer_height, 0, 0,
                cv2.BORDER_CONSTANT, value=(20, 20, 20),
            )
            lines = (
                (f"SCENE: {status} | fight score {score_text}",
                 round(31 * display_scale), 0.85, color),
                (f"CYAN BOXES: person detection only | count {len(people)}",
                 round(60 * display_scale), 0.58, person_color),
                ("Scene score does not identify individual fighters.",
                 header_height + height + round(21 * display_scale),
                 0.55, (210, 210, 210)),
            )
            for text, baseline, base_scale, text_color in lines:
                font_scale = base_scale * display_scale
                thickness = max(1, round(display_scale))
                text_width = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
                )[0][0]
                font_scale *= min(1.0, (width - 24) / max(1, text_width))
                cv2.putText(
                    canvas, text, (12, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, thickness, cv2.LINE_AA,
                )
            writer.write(canvas)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"rendered {frame_index} frames -> {output_path}")
    print("cyan boxes locate people; scene alerts do not identify individual fighters")
    return frame_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render person boxes and scene-level fight scores onto a video"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", default="models/yolo26n.pt")
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--fight-threshold", type=float, default=0.75)
    arguments = parser.parse_args()
    visualize(
        arguments.video,
        arguments.observations,
        arguments.output,
        arguments.detector,
        arguments.person_confidence,
        arguments.fight_threshold,
    )


if __name__ == "__main__":
    main()
