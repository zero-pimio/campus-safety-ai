"""Run fixed-window fall observations and durable event delivery without retraining."""
from __future__ import annotations

import argparse
import json
import signal
import time
import tomllib
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, UTC
from pathlib import Path
from threading import Event

from campus_safety_ai.adapters.easyaiot import build_platform_destination
from campus_safety_ai.adapters.fall_sources import (
    OpenCvRtspFrames, RtspOpenError, RtspReadError, RtspSourceError, cached_pose_frames,
)
from campus_safety_ai.adapters.timed_file_source import OpenCvTimedFileFrames
from campus_safety_ai.adapters.runtimes.pose_fall import FrozenPoseFallHead, SinglePersonPose, sha256
from campus_safety_ai.adapters.video_sources import redact_video_source
from campus_safety_ai.contracts import iso_time, parse_time
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import EventDelivery, JsonlDestination, OutboxCapacityError
from campus_safety_ai.core.fall_analysis import FallEventAnalysis, FallPolicy
from campus_safety_ai.core.fall_pipeline import FallPipeline, WindowPolicy
from campus_safety_ai.core.retention import register_managed_evidence
from campus_safety_ai.core.source_retry import ReconnectPolicy
from campus_safety_ai.core.storage_guard import StorageGuard, StorageGuardError, StoragePolicy
from campus_safety_ai.settings import PROJECT_ROOT, load_platform_settings


def load_policy(path: Path) -> FallPolicy:
    data = tomllib.loads(path.read_text())
    if data.pop("schema_version", None) != "1.0":
        raise ValueError("unsupported fall policy schema")
    return FallPolicy(**data)


def run(*, video: str | None = None, pose_record: Path | None = None, checkpoint: Path,
        detector: Path, output_dir: Path, camera_id: str, source_epoch: int, event_config: Path,
        started_at: datetime | None = None, device: str = "cpu", window: WindowPolicy | None = None,
        platform_config: Path | None = None, evidence: bool = True,
        reconnect: ReconnectPolicy | None = None, stop_event: Event | None = None,
        open_timeout_ms: int = 5000, read_timeout_ms: int = 5000,
        max_pending_records: int = 10000, max_pending_bytes: int = 64 * 1024 * 1024,
        storage_policy: StoragePolicy | None = None) -> dict:
    if (video is None) == (pose_record is None):
        raise ValueError("choose exactly one video or pose_record")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty")
    is_rtsp = video is not None and video.lower().startswith(("rtsp://", "rtsps://"))
    if video is not None and '://' in video and not is_rtsp:
        raise ValueError("video must be a local path or an RTSP URL")
    reconnect = reconnect or ReconnectPolicy()
    stop_event = stop_event or Event()
    started_at = started_at or datetime.now(UTC)
    iso_time(started_at)
    policy = load_policy(event_config)
    window = window or WindowPolicy()
    head = FrozenPoseFallHead(checkpoint, device)
    flow = FallPipeline(head, FallEventAnalysis(policy), window)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageGuard(output_dir.resolve(), storage_policy)
    destination = JsonlDestination(output_dir / "delivered-events.jsonl")
    delivery_type = EventDelivery
    if platform_config:
        platform = load_platform_settings(platform_config)
        destination = build_platform_destination(platform, events_path=output_dir / "delivered-events.jsonl")
        if platform.destination == "easyaiot":
            delivery_type = BackgroundDelivery
    report = {
        "status": "running", "camera_id": camera_id, "source_epoch": source_epoch,
        "started_at": iso_time(started_at), "processed_frames": 0, "observations": 0,
        "unknown_observations": 0, "starts": 0, "ends": 0, "open_events_at_exit": 0,
        "checkpoint_sha256": head.checkpoint_sha256, "model_version": head.model_version,
        "event_config_sha256": sha256(event_config), "event_policy": asdict(policy), "window": asdict(window),
        "source": str(pose_record.resolve()) if pose_record else redact_video_source(video),
        "source_kind": "verified_pose_cache" if pose_record else "rtsp" if is_rtsp else "local_video",
        "evidence_errors": [], "outbox": str((output_dir / "outbox.sqlite3").resolve()),
        "reconnect": {**asdict(reconnect), "enabled": is_rtsp, "attempts_used": 0,
                      "successful_reconnects": 0, "disconnections": 0, "open_failures": 0,
                      "read_failures": 0, "backoff_seconds_total": 0.0, "exhausted": False},
        "last_source_epoch": source_epoch, "open_timeout_ms": open_timeout_ms,
        "read_timeout_ms": read_timeout_ms,
        "delivery_limits": {"max_pending_records": max_pending_records, "max_pending_bytes": max_pending_bytes},
        "unsubmitted_records": 0, "outstanding_event_ids": [],
        "scope": "Frozen model with explicit fixed-window and event policies. No event accuracy claim without truth. "
                 "END means observed motion cleared or source/track closed, never verified human recovery.",
    }
    report["timestamp_basis"] = (
        "official cache timestamps relative to first frame" if pose_record else
        "run-level monotonic receive time anchored to UTC; capture/network latency unknown; "
        "disconnected intervals contain no observations" if is_rtsp else
        "decoded presentation timestamps from ffprobe relative to first frame; anchored to started_at"
    )
    run_path = output_dir / "run.json"
    tick, last_frame = time.monotonic(), None
    receive_anchor = (datetime.now(UTC), time.monotonic())
    active = set()
    active_evidence = {}
    outstanding = set()

    def safe_error(error):
        # Third-party backend exceptions may echo an unredacted URL. Their text
        # is deliberately omitted in RTSP reports, including model/close errors.
        return f"{type(error).__name__}: RTSP run failed" if is_rtsp else f"{type(error).__name__}: {error}"

    def save_report():
        report["elapsed_seconds"] = time.monotonic() - tick
        report["open_events_at_exit"] = len(active)
        report["outstanding_event_ids"] = sorted(outstanding)
        report["storage"] = storage.report()
        run_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    save_report()
    try:
        storage.check(0)
        with ExitStack() as stack:
            delivery = stack.enter_context(delivery_type(output_dir / "outbox.sqlite3", destination,
                                                        max_pending_records=max_pending_records,
                                                        max_pending_bytes=max_pending_bytes))
            observations_file = stack.enter_context((output_dir / "observations.jsonl").open("x"))
            events_file = stack.enter_context((output_dir / "events.jsonl").open("x"))
            unsubmitted_file = stack.enter_context((output_dir / "unsubmitted-events.jsonl").open("x"))

            def consume(batch):
                if batch.observation is not None:
                    item = {**batch.observation.to_dict(), "quality": batch.quality}
                    observations_file.write(json.dumps(item, allow_nan=False) + "\n")
                    observations_file.flush()
                    report["observations"] += 1
                    report["unknown_observations"] += item["score"] is None
                enriched = []
                for event in batch.events:
                    if evidence and last_frame is not None and event.phase == "START":
                        import cv2

                        try:
                            path = output_dir / "evidence" / f"{event.event_id}.jpg"
                            path.parent.mkdir(exist_ok=True)
                            if not cv2.imwrite(str(path), last_frame.image):
                                raise OSError("snapshot writer returned false")
                            register_managed_evidence(output_dir / "evidence", path)
                            event = replace(event, evidence_uris=(str(path.resolve()),))
                        except Exception as error:
                            if len(report["evidence_errors"]) < 100:
                                report["evidence_errors"].append(safe_error(error))
                    enriched.append(event)
                    if event.evidence_uris:
                        active_evidence[event.event_id] = event.evidence_uris
                    elif event.event_id in active_evidence:
                        enriched[-1] = replace(event, evidence_uris=active_evidence[event.event_id])
                # Audit every generated lifecycle change before queueing it.
                # If the queue rejects a batch, keep its full payload for recovery.
                for event in enriched:
                    item = {**event.to_dict(), "sourceEpoch": batch.source_epoch,
                            "reason": batch.event_reasons.get(event.idempotency_key)}
                    events_file.write(json.dumps(item, allow_nan=False) + "\n")
                    if event.phase == "START":
                        report["starts"] += 1
                        active.add(event.event_id)
                    elif event.phase == "END":
                        report["ends"] += 1
                        active.discard(event.event_id)
                        active_evidence.pop(event.event_id, None)
                events_file.flush()
                try:
                    delivery.submit(enriched)
                except Exception:
                    for event in enriched:
                        unsubmitted_file.write(json.dumps(event.to_dict(), allow_nan=False) + "\n")
                        outstanding.add(event.event_id)
                    unsubmitted_file.flush()
                    report["unsubmitted_records"] += len(enriched)
                    raise

            def run_rtsp():
                nonlocal last_frame
                epoch = source_epoch
                stats = report["reconnect"]
                while not stop_event.is_set():
                    source_error = None
                    source = None
                    perception = None
                    try:
                        try:
                            source = OpenCvRtspFrames(video, camera_id, epoch, stop_event=stop_event,
                                                     open_timeout_ms=open_timeout_ms,
                                                     read_timeout_ms=read_timeout_ms,
                                                     timestamp_anchor=receive_anchor)
                        except RtspSourceError as error:
                            source_error = error
                        if source is not None:
                            with source:
                                # A fresh association state owns exactly one source
                                # epoch; no pose/candidate survives reconnection.
                                perception = SinglePersonPose(detector, head, device)
                                frames = iter(source)
                                regained = False
                                while not stop_event.is_set():
                                    try:
                                        frame = next(frames)
                                    except StopIteration:
                                        if not stop_event.is_set():
                                            source_error = RtspReadError("RTSP source epoch ended")
                                        break
                                    except RtspSourceError as error:
                                        source_error = error
                                        break
                                    if not regained and stats["attempts_used"]:
                                        stats["successful_reconnects"] += 1
                                    regained = True
                                    last_frame = frame
                                    report["processed_frames"] += 1
                                    storage.check(report["processed_frames"])
                                    consume(flow.advance(perception.predict(frame)))
                    finally:
                        if source is not None:
                            source.close()
                        perception = None
                    if source_error is None:
                        break
                    # Finalize before waiting, including exhausted-budget exits.
                    stats["disconnections"] += 1
                    stats["open_failures" if isinstance(source_error, RtspOpenError) else "read_failures"] += 1
                    report["last_source_failure"] = type(source_error).__name__
                    consume(flow.finalize("interrupted" if stop_event.is_set() else "source_disconnected"))
                    last_frame = None
                    frame = None
                    if stop_event.is_set():
                        break
                    if stats["attempts_used"] >= reconnect.attempts:
                        stats["exhausted"] = True
                        raise RtspSourceError("RTSP reconnect budget exhausted") from None
                    delay = reconnect.delay(stats["attempts_used"] + 1)
                    report["status"] = "reconnecting"
                    save_report()
                    wait_started = time.monotonic()
                    try:
                        stopped = stop_event.wait(delay)
                    finally:
                        stats["backoff_seconds_total"] += time.monotonic() - wait_started
                    if stopped:
                        break
                    stats["attempts_used"] += 1
                    epoch += 1
                    report["last_source_epoch"] = epoch
                    report["status"] = "running"
                    save_report()

            try:
                if pose_record:
                    poses = cached_pose_frames(pose_record, head, camera_id, source_epoch, started_at)
                    for pose in poses:
                        if stop_event.is_set():
                            break
                        report["processed_frames"] += 1
                        storage.check(report["processed_frames"])
                        consume(flow.advance(pose))
                elif is_rtsp:
                    run_rtsp()
                else:
                    perception = SinglePersonPose(detector, head, device)
                    source = stack.enter_context(
                        OpenCvTimedFileFrames(Path(video), camera_id, source_epoch, started_at)
                    )
                    for frame in source:
                        if stop_event.is_set():
                            break
                        last_frame = frame
                        report["processed_frames"] += 1
                        storage.check(report["processed_frames"])
                        consume(flow.advance(perception.predict(frame)))
                consume(flow.finalize("interrupted" if stop_event.is_set() else "source_end"))
                if report["processed_frames"] == 0 and not stop_event.is_set():
                    raise ValueError("source produced no frames")
                report["pending_at_source_end"] = delivery.pending_count()
                report["delivery_status"] = delivery.retry_status()
            except BaseException as error:
                try:
                    consume(flow.finalize("interrupted" if isinstance(error, KeyboardInterrupt) else "source_error"))
                except BaseException as close_error:
                    report["finalize_error"] = safe_error(close_error)
                try:
                    report["delivery_status"] = delivery.retry_status()
                except Exception as status_error:
                    report["delivery_status_error"] = safe_error(status_error)
                raise
        report["status"] = "interrupted" if stop_event.is_set() else "completed"
    except BaseException as error:
        report["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                            "blocked" if isinstance(error, (OutboxCapacityError, StorageGuardError)) else "failed")
        # Sources deliberately raise credential-free errors; never save the input RTSP URL.
        report["error"] = safe_error(error)
        if is_rtsp and not isinstance(error, (RtspSourceError, OutboxCapacityError, StorageGuardError,
                                             KeyboardInterrupt, SystemExit)):
            raise RuntimeError("RTSP processing failed; consult the sanitized run report") from None
        raise
    finally:
        save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="local video with decoded PTS or RTSP URL (bounded reconnect)")
    source.add_argument("--pose-record", type=Path, help="verified continuous pose-cache JSON")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "runtime/training/fall-pose-dense-v1-20260921/best.pt")
    parser.add_argument("--detector", type=Path, default=PROJECT_ROOT / "models/yolo26n-pose.pt")
    parser.add_argument("--event-config", type=Path, default=PROJECT_ROOT / "configs/events/fall-v1.toml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--source-epoch", type=int, default=1)
    parser.add_argument("--started-at")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-seconds", type=float, default=1)
    parser.add_argument("--stride-seconds", type=float, default=.1)
    parser.add_argument("--platform-config", type=Path)
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--reconnect-attempts", type=int, default=5, help="total RTSP retries per run; 0 disables")
    parser.add_argument("--reconnect-initial-backoff-seconds", type=float, default=1)
    parser.add_argument("--reconnect-max-backoff-seconds", type=float, default=30)
    parser.add_argument("--rtsp-open-timeout-ms", type=int, default=5000)
    parser.add_argument("--rtsp-read-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-pending-records", type=int, default=10000)
    parser.add_argument("--max-pending-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--min-free-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-run-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--storage-check-every-frames", type=int, default=30)
    args = parser.parse_args()
    stop = Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda *_: stop.set())
    try:
        report = run(video=args.video, pose_record=args.pose_record, checkpoint=args.checkpoint,
                     detector=args.detector, event_config=args.event_config, output_dir=args.output_dir,
                     camera_id=args.camera_id, source_epoch=args.source_epoch,
                     started_at=parse_time(args.started_at) if args.started_at else None, device=args.device,
                     window=WindowPolicy(window_seconds=args.window_seconds, stride_seconds=args.stride_seconds),
                     platform_config=args.platform_config, evidence=not args.no_evidence,
                     reconnect=ReconnectPolicy(args.reconnect_attempts, args.reconnect_initial_backoff_seconds,
                                               args.reconnect_max_backoff_seconds),
                     open_timeout_ms=args.rtsp_open_timeout_ms, read_timeout_ms=args.rtsp_read_timeout_ms,
                     max_pending_records=args.max_pending_records, max_pending_bytes=args.max_pending_bytes,
                     storage_policy=StoragePolicy(args.min_free_bytes, args.max_run_bytes,
                                                  args.storage_check_every_frames),
                     stop_event=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({key: report[key] for key in ('status', 'processed_frames', 'observations', 'starts', 'ends')}))
    print(f"report -> {args.output_dir / 'run.json'}")


if __name__ == "__main__":
    main()
