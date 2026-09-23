#!/usr/bin/env python3
"""Render saved fall-run facts over original VFR frames, without running inference.

Scores and events come only from the supplied run. The pose overlay comes from
a separately registered inference cache, whose track IDs are deliberately
never mapped to the run's track IDs.
The output is a research replay, not a claim of deployment or human recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from bisect import bisect_right
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1440, 936)
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
BG, PANEL = "#0D1723", "#172536"
TEXT, MUTED = "#EDF4F8", "#ACBDCD"
TEAL, AMBER, RED = "#5DD6C4", "#F0BF6A", "#FF737B"
LINKS = ((5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8),
         (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def time_us(text: str, origin: datetime) -> int:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return round((stamp - origin).total_seconds() * 1_000_000)


def microseconds(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite source time")
    return round(float(value) * 1_000_000)


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,codec_name,time_base,duration,nb_frames,level,r_frame_rate:format=duration,size:"
        "frame=best_effort_timestamp_time,duration_time", "-of", "json", str(path),
    ], text=True))


def validate(run_dir: Path, cache_path: Path, checkpoint: Path, manifest_path: Path) -> dict:
    paths = {name: run_dir / name for name in ("run.json", "observations.jsonl", "events.jsonl")}
    run, observations, events = read(paths["run.json"]), jsonl(paths["observations.jsonl"]), jsonl(paths["events.jsonl"])
    if run["status"] != "completed" or run["source_kind"] != "local_video":
        raise ValueError("only completed local-video runs may be rendered")
    source = Path(run["source"])
    source = source if source.is_absolute() else ROOT / source
    cache = read(cache_path)
    samples = [row for row in read(manifest_path)["samples"] if row["sample_id"] == run["camera_id"]]
    if len(samples) != 1:
        raise ValueError("run camera must identify one manifest sample")
    sample = samples[0]
    if (ROOT / sample["video_path"]).resolve() != source.resolve():
        raise ValueError("manifest source path does not match run")
    source_sha = sha(source)
    if source_sha != sample["sha256"] or source_sha != cache["source_sha256"]:
        raise ValueError("source SHA differs from manifest or pose cache")
    if cache["sample_id"] != run["camera_id"]:
        raise ValueError("pose cache belongs to another camera/sample")
    identity_path, prepared_path = cache_path.parent / "identity.json", cache_path.parent / "prepared.json"
    identity, prepared = read(identity_path), read(prepared_path)
    if (cache["identity_sha256"] != sha(identity_path)
            or prepared["identity_sha256"] != sha(identity_path)
            or prepared["records"].get(cache["sample_id"]) != sha(cache_path)):
        raise ValueError("pose cache is unregistered or its identity changed")
    if sha(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match recorded run")
    if run["model_version"] != "pose-motion-" + run["checkpoint_sha256"][:16]:
        raise ValueError("model version is not bound to checkpoint hash")
    video = probe(source)
    if len(video["streams"]) != 1 or not video["frames"]:
        raise ValueError("source must contain a decodable video stream")
    source_pts = [microseconds(row["best_effort_timestamp_time"]) for row in video["frames"]]
    pts = [value - source_pts[0] for value in source_pts]
    if any(a >= b for a, b in zip(pts, pts[1:], strict=False)):
        raise ValueError("source PTS must increase strictly")
    if (len(pts) != run["processed_frames"] or len(pts) != len(cache["frames"])
            or [microseconds(x) for x in cache["timestamps_s"]] != pts
            or [microseconds(x) for x in sample["timing"]["frame_timestamps_s"]] != pts):
        raise ValueError("frame count or PTS differs between source, run, manifest and cache")
    end_us = microseconds(sample["timing"]["duration_s"])
    last_duration = microseconds(video["frames"][-1]["duration_time"])
    if last_duration <= 0 or abs(end_us - pts[-1] - last_duration) > 1:
        raise ValueError("manifest duration does not preserve the source final frame duration")
    origin = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
    if origin.tzinfo is None:
        raise ValueError("run origin must include a timezone")
    prior_sequence, prior_time = 0, -1
    for row in observations:
        seq = row["sequence"]
        if (row["cameraId"] != run["camera_id"] or row["sourceEpoch"] != run["source_epoch"]
                or row["modelVersion"] != run["model_version"] or type(seq) is not int
                or not prior_sequence < seq <= len(pts)):
            raise ValueError("observation identity or sequence mismatch")
        observed = time_us(row["observedAt"], origin)
        if (observed != pts[seq - 1] or observed < prior_time
                or not 0 <= time_us(row["windowStartedAt"], origin)
                <= time_us(row["windowEndedAt"], origin) <= observed):
            raise ValueError("observation timestamps do not match decoded source sequence")
        score = row["score"]
        if score is not None and (isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid observation score")
        for seq_input in row.get("quality", {}).get("sample_sequences", []):
            if type(seq_input) is not int or not 1 <= seq_input <= seq:
                raise ValueError("observation contains future or invalid sampled frames")
        row["media_us"] = observed
        prior_sequence, prior_time = seq, observed
    if len(observations) != run["observations"]:
        raise ValueError("observation count differs from run")
    event_tips, keys, previous_time = {}, set(), -1
    for event in events:
        observed = time_us(event["observedAt"], origin)
        if (event["cameraId"] != run["camera_id"] or event["sourceEpoch"] != run["source_epoch"]
                or event["modelVersion"] != run["model_version"]
                or event["configVersion"] != run["event_policy"]["config_version"]
                or observed not in pts or observed < previous_time
                or event["idempotencyKey"] in keys):
            raise ValueError("event identity, ordering or timestamp mismatch")
        previous = event_tips.get(event["eventId"])
        if ((previous is None and (event["revision"] != 1 or event["phase"] != "START"))
                or (previous is not None and (previous["phase"] == "END"
                                             or event["revision"] != previous["revision"] + 1))):
            raise ValueError("event lifecycle is incomplete or out of order")
        if event["phase"] not in {"START", "UPDATE", "END"}:
            raise ValueError("unsupported event phase")
        if event["phase"] == "END" and event["endedAt"] != event["observedAt"]:
            raise ValueError("this renderer requires END at its actual closure observation")
        if time_us(event["startedAt"], origin) > observed:
            raise ValueError("event starts in the future")
        event["media_us"] = observed
        event_tips[event["eventId"]] = event
        keys.add(event["idempotencyKey"])
        previous_time = observed
    phases = Counter(row["phase"] for row in events)
    if (phases["START"] != run["starts"] or phases["END"] != run["ends"]
            or any(row["phase"] != "END" for row in event_tips.values())):
        raise ValueError("event counts or closed-state claim differs from run")
    for index, pose in enumerate(cache["frames"]):
        if pose["sequence"] != index + 1:
            raise ValueError("pose-cache sequence differs from source")
        if pose["valid"]:
            box, points = pose["box"], pose["keypoints"]
            if (len(box) != 4 or len(points) != 17 or box[2] <= box[0] or box[3] <= box[1]
                    or not all(math.isfinite(v) for v in box)
                    or any(len(point) != 3 or not all(math.isfinite(v) for v in point)
                           or not 0 <= point[2] <= 1 for point in points)):
                raise ValueError("invalid pose-cache overlay geometry")
    files = {**paths, "source": source, "checkpoint": checkpoint, "manifest": manifest_path,
             "pose_cache": cache_path, "pose_identity": identity_path, "pose_registry": prepared_path}
    return {"run": run, "observations": observations, "events": events, "sample": sample,
            "cache": cache, "identity": identity, "source": source, "source_probe": video,
            "pts": pts, "end_us": end_us, "files": files,
            "hashes": {key: sha(path) for key, path in files.items()}}


@lru_cache(maxsize=24)
def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size)


def frame_state(data: dict, index: int) -> dict:
    now = data["pts"][index]
    observation_index = bisect_right([r["media_us"] for r in data["observations"]], now) - 1
    observation = data["observations"][observation_index] if observation_index >= 0 else None
    visible_events = [event for event in data["events"] if event["media_us"] <= now]
    tips = {event["eventId"]: event for event in visible_events}
    active = [event for event in tips.values() if event["phase"] != "END"]
    last_event = visible_events[-1] if visible_events else None
    return {"media_us": now, "observation": observation, "active": active,
            "last_event": last_event, "visible_events": visible_events}


def draw_frame(source: Image.Image, data: dict, index: int, state: dict) -> Image.Image:
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "跌倒识别 · 连续事件回放", font=font(38), fill=TEXT)
    draw.text((26, 72), "候选模型（非默认）  /  验证素材  /  本地回放，非板端实时", font=font(22), fill=AMBER)
    draw.text((26, 108), f"{data['sample']['dataset']} · {data['run']['camera_id']}  |  原速 · 完整源画面 · 保留 VFR 时间戳",
              font=font(17), fill=MUTED)
    x0, y0, width, height = 24, 152, 960, 540
    ratio = min(width / source.width, height / source.height)
    image = source.resize((round(source.width * ratio), round(source.height * ratio)), Image.Resampling.LANCZOS)
    left, top = x0 + (width - image.width) // 2, y0 + (height - image.height) // 2
    canvas.paste(image, (left, top))
    pose = data["cache"]["frames"][index]
    if pose["valid"]:
        box = pose["box"]
        draw.rectangle((left + box[0] * ratio, top + box[1] * ratio,
                        left + box[2] * ratio, top + box[3] * ratio), outline=TEAL, width=2)
        points = pose["keypoints"]
        for a, b in LINKS:
            if points[a][2] >= .25 and points[b][2] >= .25:
                draw.line((left + points[a][0] * ratio, top + points[a][1] * ratio,
                           left + points[b][0] * ratio, top + points[b][1] * ratio), fill=TEAL, width=3)
    draw.text((24, 704), "青色框/骨架：另一次真实推理的姿态缓存；未将缓存轨迹编号映射到本次运行。", font=font(17), fill=TEAL)
    draw.rounded_rectangle((1008, 152, 1416, 727), radius=16, fill=PANEL)
    observation, last = state["observation"], state["last_event"]
    if state["active"]:
        event_text, event_detail, color = "START · 跌倒告警", "事件仍打开", RED
    elif last and last["phase"] == "END":
        event_text, event_detail, color = "END · 事件已收口", "收口不等于人员恢复", AMBER
    else:
        event_text, event_detail, color = "尚未触发告警", "仅展示当时已产生的记录", MUTED
    draw.text((1032, 172), "实际运行事件", font=font(20), fill=MUTED)
    draw.text((1032, 213), event_text, font=font(29), fill=color)
    draw.text((1032, 259), event_detail, font=font(20), fill=color)
    reason = observation["reason"] if observation else "warming_up"
    score = observation["score"] if observation else None
    if score is None:
        state_text = "预热 · 正在收集动作" if reason == "warming_up" else "不确定 · 观测不完整"
        state_color = AMBER
    elif state["active"]:
        state_text, state_color = "跌倒动作观测", RED
    elif last and last["phase"] == "END":
        state_text, state_color = "当前动作观测 · 不推断恢复", MUTED
    elif score >= data["run"]["event_policy"]["start_score"]:
        state_text, state_color = "候选 · 等待持续确认", AMBER
    else:
        state_text, state_color = "正常观测 · 未触发告警", TEAL
    draw.line((1032, 305, 1392, 305), fill="#304258", width=1)
    draw.text((1032, 325), state_text, font=font(21), fill=state_color)
    draw.text((1032, 365), "当前窗口分数", font=font(20), fill=MUTED)
    draw.text((1032, 396), "—" if score is None else f"{score:.3f}", font=font(42), fill=state_color)
    draw.text((1174, 417), f"启动阈值 {data['run']['event_policy']['start_score']:.2f}", font=font(18), fill=MUTED)
    draw.text((1032, 467), f"观测原因：{reason}", font=font(17), fill=MUTED)
    seq = observation["sequence"] if observation else "—"
    observed = observation["media_us"] / 1_000_000 if observation else 0
    draw.text((1032, 499), f"观测帧 {seq}  /  产生于 {observed:.6f} s", font=font(17), fill=MUTED)
    draw.line((1032, 539, 1392, 539), fill="#304258", width=1)
    if last:
        draw.text((1032, 559), f"最近记录：{last['phase']} @ {last['media_us'] / 1_000_000:.6f} s",
                  font=font(19), fill=color)
        draw.text((1032, 594), f"原因：{last.get('reason', 'unspecified')}", font=font(19), fill=color)
        if last["phase"] == "END":
            closure = "轨迹改变触发关闭；" if last.get("reason") == "track_changed" else "结束的是事件记录；"
            draw.text((1032, 635), closure, font=font(20), fill=AMBER)
            draw.text((1032, 668), "收口未证明人员恢复。", font=font(20), fill=AMBER)
    else:
        draw.text((1032, 562), "正常/低分 ≠ 人员安全", font=font(20), fill=MUTED)
        draw.text((1032, 603), "事件仅按 observedAt 出现，", font=font(19), fill=MUTED)
        draw.text((1032, 635), "不按 startedAt 提前回填。", font=font(19), fill=MUTED)
    draw.rounded_rectangle((24, 753, 1416, 912), radius=16, fill=PANEL)
    current = state["media_us"] / 1_000_000
    draw.text((44, 768), f"素材时间 {current:.6f} s  /  {data['end_us'] / 1_000_000:.6f} s"
              f"       原始帧 {index + 1} / {len(data['pts'])}", font=font(23), fill=TEXT)
    draw.text((44, 811), "分数与 START/END 来自现成运行日志；姿态来自独立缓存，仅用于画面定位。", font=font(20), fill=MUTED)
    draw.text((44, 847), "没有正常→跌倒→真实恢复的完整素材。本片不证明恢复、不代表校园验收通过。", font=font(20), fill=AMBER)
    draw.text((44, 881), f"{data['run']['model_version']}  |  {data['run']['event_policy']['config_version']}", font=font(16), fill=MUTED)
    return canvas


def render(args) -> dict:
    output = args.output.resolve()
    artifacts = {"video": output, "preview": output.with_name(output.stem + "-preview.png"),
                 "end_preview": output.with_name(output.stem + "-end.png"),
                 "timeline": output.with_name(output.stem + "-timeline.jsonl"),
                 "verification": output.with_name(output.stem + "-verification.json")}
    if any(path.exists() for path in artifacts.values()):
        raise ValueError("output artifacts must not already exist; originals are never overwritten")
    if not FONT_PATH.is_file() or not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        raise RuntimeError("Chinese font, ffmpeg and ffprobe are required")
    data = validate(args.run_dir, args.pose_cache, args.checkpoint, args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline, preview_saved, end_saved = [], False, False
    with tempfile.TemporaryDirectory(prefix="fall-run-render-") as temporary:
        directory = Path(temporary)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(data["source"]),
                        "-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base", "1:1000000", "-start_number", "0",
                        str(directory / "source-%06d.png")], check=True)
        sources = sorted(directory.glob("source-*.png"))
        if len(sources) != len(data["pts"]):
            raise ValueError("complete source decoding did not preserve frame count")
        concat = ["ffconcat version 1.0"]
        for index, source in enumerate(sources):
            state = frame_state(data, index)
            with Image.open(source) as image:
                rendered = draw_frame(image.convert("RGB"), data, index, state)
            frame_path = directory / f"rendered-{index:06d}.png"
            rendered.save(frame_path)
            concat.extend([f"file '{frame_path.name}'", "option framerate 1000000"])
            if index + 1 < len(sources):
                concat.append(f"duration {(data['pts'][index + 1] - data['pts'][index]) / 1_000_000:.6f}")
            if state["active"] and not preview_saved:
                rendered.save(artifacts["preview"])
                preview_saved = True
            if state["last_event"] and state["last_event"]["phase"] == "END" and not end_saved:
                rendered.save(artifacts["end_preview"])
                end_saved = True
            observation = state["observation"]
            timeline.append({"frame_number": index + 1, "source_pts_us": data["pts"][index],
                             "observation_sequence": observation["sequence"] if observation else None,
                             "observation_observed_us": observation["media_us"] if observation else None,
                             "score": observation["score"] if observation else None,
                             "observation_reason": observation["reason"] if observation else None,
                             "visible_event_keys": [r["idempotencyKey"] for r in state["visible_events"]],
                             "active_event_ids": [r["eventId"] for r in state["active"]]})
        if not preview_saved:
            rendered.save(artifacts["preview"])
        if not end_saved:
            rendered.save(artifacts["end_preview"])
        concat_path = directory / "frames.ffconcat"
        concat_path.write_text("\n".join(concat) + "\n", encoding="utf-8")
        last_duration = data["end_us"] - data["pts"][-1]
        # PNG time base is one microsecond. Set only the final encoded packet's
        # duration, preserving that real source frame without adding a held frame.
        duration_filter = f"setts=duration=if(eq(N\\,{len(sources) - 1})\\,{last_duration}\\,DURATION)"
        # The PNG demuxer uses a microsecond clock, not a million-frame video.
        # Keep packet PTS while writing browser-compatible H.264 VUI metadata.
        video_filters = duration_filter + ",h264_metadata=level=4.1:tick_rate=60/1:fixed_frame_rate_flag=0"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-f", "concat", "-safe", "0",
                        "-i", str(concat_path), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                        "-bf", "0", "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-enc_time_base", "1:1000000",
                        "-video_track_timescale", "1000000", "-bsf:v", video_filters,
                        "-movflags", "+faststart", "-threads", "2", "-n", str(output)], check=True)
    encoded = probe(output)
    rate_n, rate_d = map(int, encoded["streams"][0]["r_frame_rate"].split("/"))
    if encoded["streams"][0]["level"] != 41 or rate_d <= 0 or rate_n / rate_d > 120:
        raise ValueError("encoded H.264 metadata is unsuitable for this browser demo")
    encoded_pts = [microseconds(row["best_effort_timestamp_time"]) for row in encoded["frames"]]
    if len(encoded_pts) != len(data["pts"]):
        raise ValueError("encoded frame count changed")
    max_error = max(abs(a - b) for a, b in zip(data["pts"], encoded_pts, strict=True))
    encoded_end = microseconds(encoded["streams"][0]["duration"])
    if max_error > 1000 or abs(encoded_end - data["end_us"]) > 1000:
        raise ValueError("encoded PTS or video-stream duration differs by more than 1 ms")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(output),
                    "-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base", "1:1000000",
                    "-f", "null", "-"], check=True)
    for key, path in data["files"].items():
        if sha(path) != data["hashes"][key]:
            raise ValueError("an input changed during rendering")
    event_displays = []
    for event in data["events"]:
        first = next(row for row in timeline if event["idempotencyKey"] in row["visible_event_keys"])
        if first["source_pts_us"] != event["media_us"]:
            raise ValueError("event display does not begin at its real observedAt")
        event_displays.append({"phase": event["phase"], "reason": event.get("reason"),
                               "event_id": event["eventId"], "idempotency_key": event["idempotencyKey"],
                               "observed_at": event["observedAt"], "started_at": event["startedAt"],
                               "first_display_frame": first["frame_number"],
                               "first_display_seconds": first["source_pts_us"] / 1_000_000})
    for row in timeline:
        if row["observation_observed_us"] is not None and row["observation_observed_us"] > row["source_pts_us"]:
            raise ValueError("a future observation appeared early")
    artifacts["timeline"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timeline), encoding="utf-8")
    report = {"status": "verified", "source": str(data["source"].resolve()), "source_sha256": data["hashes"]["source"],
              "camera_id": data["run"]["camera_id"], "source_epoch": data["run"]["source_epoch"],
              "model_version": data["run"]["model_version"], "checkpoint_sha256": data["run"]["checkpoint_sha256"],
              "sample_split": data["sample"]["split"], "model_status": "candidate, not default",
              "source_frame_count": len(data["pts"]), "output_frame_count": len(encoded_pts),
              "source_pts_us": data["pts"], "output_pts_us": encoded_pts,
              "max_pts_error_us": max_error, "source_presentation_end_seconds": data["end_us"] / 1_000_000,
              "output_video_duration_seconds": encoded_end / 1_000_000,
              "source_stream_duration_seconds": float(data["source_probe"]["streams"][0]["duration"]),
              "source_container_duration_seconds": float(data["source_probe"]["format"]["duration"]),
              "duration_basis": "last source PTS plus original final frame duration; audio not retained",
              "codec": encoded["streams"][0]["codec_name"], "output_size": list(SIZE),
              "h264_level": encoded["streams"][0]["level"],
              "nominal_frame_rate": encoded["streams"][0]["r_frame_rate"],
              "full_decode_passed": True, "causal_display_verified": True, "event_displays": event_displays,
              "observation_count": len(data["observations"]), "event_count": len(data["events"]),
              "pose_overlay": {"source": "separate registered real inference cache; localization only",
                               "identity": data["identity"], "track_identity_mapping_applied": False,
                               "cache_track_keys": sorted({r["track_key"] for r in data["cache"]["frames"] if r["track_key"]}),
                               "run_track_keys": sorted({key for r in data["observations"] for key in r["subjectTrackKeys"]})},
              "limitations": ["Local saved-run replay, not board real-time processing or campus acceptance",
                              "START is displayed at observedAt, never retrospectively at startedAt",
                              "END track_changed closes the event; person remains on floor, no recovery claim",
                              "No synthetic normal-fall-recovery sequence, score smoothing or frame repetition"],
              "input_hashes": {key: {"path": str(path.resolve()), "sha256": data["hashes"][key]}
                               for key, path in data["files"].items()},
              "renderer_sha256": sha(Path(__file__)),
              "artifacts": {key: {"path": str(path), "sha256": sha(path)}
                            for key, path in artifacts.items() if key != "verification"}}
    artifacts["verification"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/fall-web-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args)
    print(json.dumps({key: result[key] for key in ("status", "source_frame_count", "output_frame_count",
          "max_pts_error_us", "output_video_duration_seconds", "event_displays")}, ensure_ascii=False, indent=2))
    print(f"video -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
