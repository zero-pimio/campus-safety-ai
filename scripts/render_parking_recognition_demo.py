#!/usr/bin/env python3
"""Render actual, same-frame scene-video detections without inferring parking events.

This renderer accepts completed, full-frame-rate CFR local replays only. It does
not run a detector, interpolate boxes, reuse old detections, or display tracking
IDs. Its relative video timeline follows scene_video's declared frame/FPS basis;
it does not invent missing original capture PTS in legacy AVI source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
LABELS = {"car": "汽车", "truck": "货车", "bus": "客车", "motorcycle": "摩托车"}
THRESHOLD = 0.4
WIDTH, HEADER, FOOTER = 1600, 112, 76
BG, TEXT, MUTED, GREEN = "#101923", "#F1F5F8", "#B7C6D2", "#46E5B0"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


def local_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def probe(path: Path, *, count: bool = False) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        command.append("-count_frames")
    command += ["-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,"
                "r_frame_rate,nb_frames,nb_read_frames,duration:format=duration,size",
                "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command, text=True))


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(run_dir: Path) -> dict:
    run_path, detection_path = run_dir / "run.json", run_dir / "detections.jsonl"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("finish_reason") != "eof":
        raise ValueError("renderer requires a completed local replay ending at source EOF")
    if run.get("event_kind") != "parking" or run.get("frame_stride") != 1:
        raise ValueError("renderer requires a parking scene replay with frame_stride=1")
    if not str(run.get("timestamp_basis", "")).startswith("CFR file frame index / metadata FPS"):
        raise ValueError("run must explicitly declare its CFR frame-index time basis")
    source, detector = local_path(run["source"]), local_path(run["detector"])
    source_hash, detector_hash = sha(source), sha(detector)
    if source_hash != run["source_sha256"] or detector_hash != run["detector_sha256"]:
        raise ValueError("source or detector hash differs from the completed run")
    source_probe = probe(source)
    streams = source_probe["streams"]
    if len(streams) != 1:
        raise ValueError("source must contain one selected video stream")
    stream = streams[0]
    rate = Fraction(stream["avg_frame_rate"])
    fps = float(rate)
    count = run["decoded_frames"]
    if (not finite_number(run["source_fps"]) or fps <= 0
            or abs(fps - run["source_fps"]) > 1e-6
            or Fraction(stream["r_frame_rate"]) != rate):
        raise ValueError("run frame rate differs from source CFR metadata")
    if (type(count) is not int or count <= 0 or run["processed_frames"] != count
            or int(stream["nb_frames"]) != count):
        raise ValueError("source and run frame counts disagree or the replay is incomplete")
    duration = count / fps
    if abs(float(source_probe["format"]["duration"]) - duration) > 1 / fps:
        raise ValueError("source duration disagrees with the declared CFR frame timeline")
    width, height = stream["width"], stream["height"]
    origin = stamp(run["started_at"])
    rows = []
    for line in detection_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence = len(rows) + 1
        if (type(row.get("sequence")) is not int or row["sequence"] != sequence
                or row["cameraId"] != run["camera_id"]
                or row["sourceEpoch"] != run["source_epoch"]
                or row["modelVersion"] != detector.stem
                or row["width"] != width or row["height"] != height):
            raise ValueError(f"detection identity, dimensions or sequence mismatch at frame {sequence}")
        relative = (stamp(row["capturedAt"]) - origin).total_seconds()
        if abs(relative - (sequence - 1) / fps) > 1.1e-6:
            raise ValueError(f"detection time differs from source frame at sequence {sequence}")
        if not isinstance(row["detections"], list):
            raise ValueError("detections must be a list")
        for detection in row["detections"]:
            score, box = detection["confidence"], detection["bboxXyxy"]
            if (not finite_number(score) or not 0 <= score <= 1 or len(box) != 4
                    or not all(finite_number(value) for value in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height
                    or not isinstance(detection["label"], str)):
                raise ValueError(f"invalid detection geometry or confidence at frame {sequence}")
        rows.append(row)
    if len(rows) != count:
        raise ValueError("exactly one detection record per source frame is required")
    return {"run": run, "rows": rows, "source": source, "detector": detector,
            "source_probe": source_probe, "rate": rate, "fps": fps, "count": count,
            "duration": duration, "width": width, "height": height,
            "files": {"run": run_path, "detections": detection_path,
                      "source": source, "detector": detector},
            "hashes": {"run": sha(run_path), "detections": sha(detection_path),
                       "source": source_hash, "detector": detector_hash}}


@lru_cache(maxsize=16)
def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size)


def clock_text(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def visible_detections(row: dict) -> list[dict]:
    # Paint stronger same-frame evidence last when the detector returns overlapping
    # boxes. Keep all boxes; the renderer does not run an unrecorded NMS pass.
    return sorted((item for item in row["detections"]
                   if item["label"] in LABELS and item["confidence"] >= THRESHOLD),
                  key=lambda item: item["confidence"])


def draw_frame(source: Image.Image, data: dict, index: int) -> Image.Image:
    video_height = round(data["height"] * WIDTH / data["width"] / 2) * 2
    canvas = Image.new("RGB", (WIDTH, HEADER + video_height + FOOTER), BG)
    image = source.resize((WIDTH, video_height), Image.Resampling.LANCZOS)
    canvas.paste(image, (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), "车辆识别 · 新素材实测", font=font(40), fill=TEXT)
    draw.text((26, 72), "完整源画面  ·  原速播放  ·  每一帧独立推理", font=font(22), fill=MUTED)
    timer = f"{clock_text(index / data['fps'])} / {clock_text(data['duration'])}"
    timer_width = draw.textlength(timer, font=font(28))
    draw.text((WIDTH - 24 - timer_width, 24), timer, font=font(28), fill=TEXT)
    model = f"{data['detector'].stem.upper()}  ·  {data['fps']:g} fps 素材 / 本地回放"
    draw.text((WIDTH - 24 - draw.textlength(model, font=font(20)), 73), model,
              font=font(20), fill=MUTED)
    scale_x, scale_y = WIDTH / source.width, video_height / source.height
    for item in visible_detections(data["rows"][index]):
        x1, y1, x2, y2 = item["bboxXyxy"]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = HEADER + y1 * scale_y, HEADER + y2 * scale_y
        draw.rectangle((x1, y1, x2, y2), outline=GREEN, width=3)
        label = f"{LABELS[item['label']]}  {item['confidence']:.2f}"
        label_width = math.ceil(draw.textlength(label, font=font(24))) + 18
        label_left = min(x1, WIDTH - label_width)
        label_top = max(HEADER, y1 - 36)
        draw.rounded_rectangle((label_left, label_top, label_left + label_width, label_top + 34),
                               radius=4, fill=BG, outline=GREEN, width=1)
        draw.text((label_left + 9, label_top + 1), label, font=font(24), fill=GREEN)
    footer_top = HEADER + video_height
    draw.text((24, footer_top + 6), "车辆检测，不等于违停判定", font=font(26), fill=TEXT)
    draw.text((26, footer_top + 44), "只显示本帧置信度 ≥ 0.40 的车辆框；遮挡或漏检时不沿用旧框。",
              font=font(19), fill=MUTED)
    frame_text = f"帧 {index + 1} / {data['count']}"
    draw.text((WIDTH - 24 - draw.textlength(frame_text, font=font(21)), footer_top + 19),
              frame_text, font=font(21), fill=MUTED)
    return canvas


def read_frame(pipe, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def render(data: dict, output: Path) -> dict:
    output = output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must be an .mp4 file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [output, output.parent / "preview.jpg", output.parent / "verification.json"]
    if any(path.exists() for path in artifacts):
        raise ValueError("output, preview or verification already exists; choose a fresh directory")
    video_height = round(data["height"] * WIDTH / data["width"] / 2) * 2
    size = (WIDTH, HEADER + video_height + FOOTER)
    preview_index = min(data["count"] - 1, round(5 * data["fps"]))
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        rendered, preview = temporary / output.name, temporary / "preview.jpg"
        decoder = encoder = None
        with (temporary / "decode.log").open("wb") as decode_log, \
                (temporary / "encode.log").open("wb") as encode_log:
            try:
                decoder = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(data["source"]),
                    "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE, stderr=decode_log)
                encoder = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(data["rate"]),
                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(rendered)],
                    stdin=subprocess.PIPE, stderr=encode_log)
                frame_size = data["width"] * data["height"] * 3
                assert decoder.stdout is not None and encoder.stdin is not None
                for index in range(data["count"]):
                    raw = read_frame(decoder.stdout, frame_size)
                    if len(raw) != frame_size:
                        raise ValueError(f"source ended before recorded frame {index + 1}")
                    source = Image.frombytes("RGB", (data["width"], data["height"]), raw)
                    canvas = draw_frame(source, data, index)
                    if index == preview_index:
                        canvas.save(preview, quality=94)
                    encoder.stdin.write(canvas.tobytes())
                if decoder.stdout.read(1):
                    raise ValueError("source has frames beyond the recorded complete run")
                decoder.stdout.close()
                encoder.stdin.close()
                if decoder.wait() or encoder.wait():
                    raise RuntimeError("ffmpeg decode/encode failed")
            except BaseException as error:
                for process in (decoder, encoder):
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait()
                decode_log.flush()
                encode_log.flush()
                details = "\n".join((temporary / name).read_text(errors="replace")[-2000:]
                                    for name in ("decode.log", "encode.log"))
                raise RuntimeError(f"render failed: {error}\n{details}") from error
        result = probe(rendered, count=True)
        stream = result["streams"][0]
        if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
                or int(stream["nb_read_frames"]) != data["count"]
                or abs(float(result["format"]["duration"]) - data["duration"]) > 0.0011
                or (stream["width"], stream["height"]) != size
                or Fraction(stream["avg_frame_rate"]) != data["rate"]):
            raise ValueError("encoded video failed codec, timing or decoded frame-count verification")
        # Ensure saved facts and source/model files stayed unchanged during rendering.
        if any(sha(path) != data["hashes"][name] for name, path in data["files"].items()):
            raise ValueError("input provenance changed during rendering")
        counts = Counter(item["label"] for row in data["rows"] for item in visible_detections(row))
        report = {
            "status": "verified", "scope": "Vehicle recognition replay; not parking event or accuracy validation",
            "source": str(data["source"]), "run_dir": str(data["files"]["run"].parent),
            "input_sha256": data["hashes"], "renderer_sha256": sha(Path(__file__)),
            "output": str(output), "output_sha256": sha(rendered),
            "preview": str(artifacts[1]), "preview_source_frame_index": preview_index,
            "preview_sha256": sha(preview), "width": size[0], "height": size[1],
            "source_frame_count": data["count"], "output_decoded_frames": int(stream["nb_read_frames"]),
            "duration_seconds": data["duration"], "fps": str(data["rate"]),
            "timeline_basis": data["run"]["timestamp_basis"], "output_codec": stream["codec_name"],
            "output_pixel_format": stream["pix_fmt"], "confidence_threshold": THRESHOLD,
            "vehicle_labels": LABELS, "rendered_detection_boxes_by_label": dict(counts),
            "same_frame_detections_only": True, "reused_detection_boxes": 0,
            "source_frames_cropped": False, "tracking_ids_displayed": False,
            "parking_events_displayed": False, "playback_speed": 1.0,
        }
        rendered.rename(output)
        preview.rename(artifacts[1])
        artifacts[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    global FONT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh output MP4 path")
    parser.add_argument("--font", type=Path, default=FONT_PATH, help="Chinese TrueType/OpenType font")
    args = parser.parse_args()
    FONT_PATH = args.font.resolve()
    if not FONT_PATH.is_file():
        parser.error("Chinese font does not exist; pass --font")
    report = render(validate(args.run_dir.resolve()), args.output)
    print(json.dumps({key: report[key] for key in ("status", "output", "preview", "duration_seconds",
                                                  "output_decoded_frames")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
