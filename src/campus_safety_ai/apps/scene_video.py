from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.adapters.frame_sources import OpenCvFileFrames
from campus_safety_ai.adapters.runtimes.tracker_factory import build_tracker
from campus_safety_ai.adapters.runtimes.ultralytics import UltralyticsPerception
from campus_safety_ai.contracts import EventRecord, FramePacket, iso_time, parse_time
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_analysis import EventAnalysis
from campus_safety_ai.core.event_delivery import EventDelivery, JsonlDestination
from campus_safety_ai.core.parking_analysis import ParkingAnalysis
from campus_safety_ai.core.scene_pipeline import SceneFrameResult, SceneVideoPipeline
from campus_safety_ai.settings import (
    PROJECT_ROOT, load_intrusion_policy, load_parking_policy, load_platform_settings,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_records(
    records: tuple[EventRecord, ...], frame: FramePacket | None, directory: Path,
) -> tuple[list[EventRecord], list[dict[str, str]]]:
    import cv2

    result = []
    errors = []
    for record in records:
        if frame is None:
            result.append(record)
            continue
        snapshot = directory / f"{record.event_id}-{record.revision}.jpg"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(snapshot), frame.image):
                raise OSError("snapshot writer returned false")
            record = replace(record, evidence_uris=(str(snapshot.resolve()),))
        except Exception as error:
            errors.append({"idempotency_key": record.idempotency_key, "error": str(error)})
        result.append(record)
    return result, errors


def _render(
    result: SceneFrameResult, polygon: tuple[tuple[float, float], ...],
    active_count: int, event_kind: str,
) -> Any:
    import cv2
    import numpy as np

    assert result.frame is not None and result.detections is not None
    frame = result.frame.image.copy()
    width, height = result.frame.width, result.frame.height
    # Limit preview size without affecting detector coordinates or rule inputs.
    ratio = min(1.0, 960 / width)
    if ratio < 1:
        frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
    width, height = frame.shape[1], frame.shape[0]
    points = np.array([(round(x * (width - 1)), round(y * (height - 1))) for x, y in polygon])
    cv2.polylines(frame, [points.astype(np.int32)], True, (0, 220, 255), 2)
    for detection in result.detections.detections:
        box = detection.bbox
        left, top, right, bottom = [round(value * ratio) for value in (box.x1, box.y1, box.x2, box.y2)]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 210, 40), 2)
        cv2.circle(frame, ((left + right) // 2, bottom), 4, (255, 210, 40), -1)
        cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}", (left, max(14, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 210, 40), 1, cv2.LINE_AA)
    color = (40, 40, 245) if active_count else (190, 190, 190)
    if active_count:
        cv2.rectangle(frame, (2, 2), (width - 3, height - 3), color, 4)
    frame = cv2.copyMakeBorder(frame, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(frame, f"{event_kind.upper()} | active events: {active_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Demo zone (yellow) | detections (cyan) | frame bottom point", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, min(0.45, width / 1400), (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def run(
    video: Path, output_dir: Path, *, event_kind: str, event_config: Path,
    scene_config: Path, detector: Path, camera_id: str, source_epoch: int = 1,
    started_at: datetime | None = None, frame_stride: int = 3,
    tracker_backend: str = "bytetrack", detection_confidence: float = 0.1,
    device: str = "cpu", platform_config: Path | None = None, annotated_video: bool = True,
) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("scene replay output directory must be empty; use a new directory")
    if not video.is_file() or not detector.is_file():
        raise ValueError("video and detector must be existing local files; this command does not download weights")
    if event_kind not in {"intrusion", "parking"}:
        raise ValueError("event_kind must be intrusion or parking")
    if not math.isfinite(detection_confidence) or not 0 <= detection_confidence <= 1:
        raise ValueError("detection_confidence must be finite and between zero and one")
    policy = (
        load_intrusion_policy(event_config, scene_config) if event_kind == "intrusion"
        else load_parking_policy(event_config, scene_config)
    )
    started_at = started_at or datetime.now(UTC)
    source = OpenCvFileFrames(video, camera_id, source_epoch, started_at, frame_stride)
    with source:
        if frame_stride / source.fps > policy.max_observation_gap_seconds:
            raise ValueError("frame stride exceeds the policy observation-gap limit")
        output_dir.mkdir(parents=True, exist_ok=True)
        events_path = output_dir / "events.jsonl"
        destination = JsonlDestination(events_path)
        delivery_type = EventDelivery
        if platform_config is not None:
            platform = load_platform_settings(platform_config)
            destination = build_platform_destination(platform, events_path=events_path)
            if platform.destination == "easyaiot":
                delivery_type = BackgroundDelivery
        # A replay always owns its fresh outbox; never mix demo traffic into a live database.
        outbox_path = output_dir / "outbox.sqlite3"
        report: dict[str, Any] = {
            "status": "running", "event_kind": event_kind,
            "source": str(video.resolve()), "source_sha256": _sha256(video),
            "detector": str(detector.resolve()), "detector_sha256": _sha256(detector),
            "event_config": str(event_config.resolve()), "event_config_sha256": _sha256(event_config),
            "scene_config": str(scene_config.resolve()), "scene_config_sha256": _sha256(scene_config),
            "camera_id": camera_id, "source_epoch": source_epoch, "started_at": iso_time(started_at),
            "source_fps": source.fps, "frame_stride": frame_stride, "tracker": tracker_backend,
            "detection_confidence": detection_confidence, "device": device,
            "timestamp_basis": "CFR file frame index / metadata FPS; not RTSP wall-clock timing",
            "processed_frames": 0, "event_records": 0, "starts": 0, "ends": 0,
            "evidence_errors": [], "outbox": str(outbox_path.resolve()),
            "scope": "Local replay with configured zones; no annotated scene accuracy or deployment acceptance claim",
        }
        run_path = output_dir / "run.json"
        run_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        writer = None
        last_frame = None
        active: set[str] = set()
        tick = time.perf_counter()

        def consume_records(records: tuple[EventRecord, ...] | list[EventRecord], delivery: Any) -> None:
            enriched, errors = _snapshot_records(tuple(records), last_frame, output_dir / "evidence")
            report["evidence_errors"].extend(errors)
            # submit persists the complete batch before attempting delivery.
            delivery.submit(enriched)
            for record in enriched:
                report["event_records"] += 1
                if record.phase == "START":
                    report["starts"] += 1
                    active.add(record.event_id)
                elif record.phase == "END":
                    report["ends"] += 1
                    active.discard(record.event_id)

        try:
            tracker = build_tracker(tracker_backend)
            analysis = EventAnalysis(policy, tracker) if event_kind == "intrusion" else ParkingAnalysis(policy, tracker)
            perception = UltralyticsPerception(
                str(detector), detector.stem, confidence=detection_confidence, device=device,
            )
            pipeline = SceneVideoPipeline(perception, analysis)
            with delivery_type(outbox_path, destination) as delivery:
                try:
                    with (output_dir / "detections.jsonl").open("x") as detections_file:
                        for batch in pipeline.stream(source):
                            if batch.frame is not None:
                                last_frame = batch.frame
                                report["processed_frames"] += 1
                            consume_records(batch.events, delivery)
                            if batch.detections is not None:
                                detections_file.write(json.dumps(batch.detections.to_dict()) + "\n")
                                detections_file.flush()
                                if annotated_video:
                                    import cv2

                                    rendered = _render(batch, policy.polygon, len(active), event_kind)
                                    if writer is None:
                                        writer = cv2.VideoWriter(
                                            str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            source.fps / frame_stride, (rendered.shape[1], rendered.shape[0]),
                                        )
                                        if not writer.isOpened():
                                            raise OSError("cannot create annotated replay")
                                    writer.write(rendered)
                            if batch.finish_reason is not None:
                                report["finish_reason"] = batch.finish_reason
                except BaseException:
                    # Decoder errors already finalize in the pipeline. Renderer,
                    # consumer and interrupt failures must also close before the
                    # delivery context exits. Never replace the original error.
                    if last_frame is not None:
                        try:
                            consume_records(analysis.finalize(
                                last_frame.camera_id, last_frame.source_epoch, last_frame.captured_at,
                            ), delivery)
                        except BaseException as close_error:
                            report["finalize_error"] = f"{type(close_error).__name__}: {close_error}"
                    raise
            if report["processed_frames"] == 0:
                raise ValueError("video contained no decoded frames")
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if writer is not None:
                writer.release()
            report["decoded_frames"] = source.decoded_frames
            report["elapsed_seconds"] = time.perf_counter() - tick
            report["open_events_at_exit"] = len(active)
            run_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay local video through detection, intrusion/parking and outbox")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event", choices=("intrusion", "parking"), default="intrusion")
    parser.add_argument("--scene-config", required=True, type=Path, help="explicit zones for this video")
    parser.add_argument("--event-config", type=Path)
    parser.add_argument("--detector", type=Path, default=PROJECT_ROOT / "models/yolo26n.pt")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--source-epoch", type=int, default=1)
    parser.add_argument("--started-at")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--tracker", choices=("simple_iou", "bytetrack"), default="bytetrack")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--platform-config", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    result = run(
        args.video, args.output_dir, event_kind=args.event,
        event_config=args.event_config or PROJECT_ROOT / f"configs/events/{args.event}-v1.toml",
        scene_config=args.scene_config, detector=args.detector, camera_id=args.camera_id,
        source_epoch=args.source_epoch, started_at=parse_time(args.started_at) if args.started_at else None,
        frame_stride=args.frame_stride, tracker_backend=args.tracker, detection_confidence=args.confidence,
        device=args.device, platform_config=args.platform_config, annotated_video=not args.no_video,
    )
    print(json.dumps({key: result[key] for key in ("status", "processed_frames", "starts", "ends")}, indent=2))
    print(f"report -> {args.output_dir / 'run.json'}")


if __name__ == "__main__":
    main()
