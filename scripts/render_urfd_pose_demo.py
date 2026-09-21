"""Render verified continuous-frame model predictions without altering their scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1280, 800)
COLORS = {"warming_up": "#EBB95E", "unknown": "#EBB95E",
          "no_fall_transition": "#35D6B3", "fall_transition": "#FF655D"}
LABELS = {"warming_up": "正在收集动作", "unknown": "暂时无法判断",
          "no_fall_transition": "未见跌倒动作", "fall_transition": "检测到跌倒动作"}
FONT = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def draw_frame(path: Path, sequence: str, row: dict, pose: dict, history: list[dict], total_ms: float) -> Image.Image:
    canvas = Image.new("RGB", SIZE, "#101923")
    draw = ImageDraw.Draw(canvas)
    fonts = {size: ImageFont.truetype(str(FONT), size) for size in (17, 19, 23, 28, 34)}
    title = "日常走路" if sequence == "adl-10" else "跌倒片段"
    draw.text((24, 10), f"{title} · 新版模型识别回放", font=fonts[34], fill="white")
    draw.text((26, 53), "真实连续画面 / 原速播放 / 只使用已经出现的画面进行判断", font=fonts[19], fill="#B9C9D7")
    with Image.open(path) as source:
        canvas.paste(source.convert("RGB").resize((960, 720), Image.Resampling.LANCZOS), (0, 80))
    color = COLORS[row["prediction"]]
    index = row["frame_index"]
    tracking = pose["tracking"][index]
    if tracking["association_valid"]:
        box = pose["boxes"][index]
        draw.rectangle((box[0] * 1.5, 80 + box[1] * 1.5, box[2] * 1.5, 80 + box[3] * 1.5), outline=color, width=3)
        points = pose["keypoints"][index]
        for a, b in ((5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)):
            if points[a][2] >= .25 and points[b][2] >= .25:
                draw.line((points[a][0] * 1.5, 80 + points[a][1] * 1.5, points[b][0] * 1.5, 80 + points[b][1] * 1.5), fill=color, width=3)
        for x, y, confidence in points:
            if confidence >= .25:
                x, y = x * 1.5, 80 + y * 1.5
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    # Keep all explanation outside the original camera view.
    draw.rectangle((960, 80, 1280, 800), fill="#101923")
    draw.text((984, 104), "当前模型判断", font=fonts[19], fill="#B9C9D7")
    draw.rounded_rectangle((978, 143, 1264, 207), radius=9, fill="#202F3D")
    draw.text((990, 156), LABELS[row["prediction"]], font=fonts[28], fill=color)
    score = row["score"]
    draw.text((984, 235), "跌倒动作分数", font=fonts[19], fill="#B9C9D7")
    draw.text((984, 271), "—" if score is None else f"{score:.3f}", font=fonts[34], fill="white")
    draw.rounded_rectangle((984, 326, 1256, 336), radius=4, fill="#32404D")
    if score is not None and score > 0:
        draw.rectangle((984, 326, 984 + 272 * score, 336), fill=color)
    draw.line((1120, 319, 1120, 343), fill="white", width=2)
    draw.text((984, 350), "判断阈值 0.500", font=fonts[17], fill="#B9C9D7")
    draw.text((984, 410), "已经观察到的分数", font=fonts[19], fill="#B9C9D7")
    draw.rectangle((984, 452, 1256, 572), fill="#192633")
    draw.line((984, 512, 1256, 512), fill="#607485", width=1)
    curve = [(984 + 272 * r["source_timestamp_ms"] / total_ms, 572 - 120 * r["score"])
             for r in history if r["score"] is not None]
    if len(curve) > 1:
        draw.line(curve, fill=color, width=2)
    draw.text((984, 593), f"素材时间 {row['source_timestamp_ms'] / 1000:.3f} 秒", font=fonts[19], fill="white")
    draw.text((984, 626), f"第 {row['frame_number']} 帧 · {sequence}", font=fonts[19], fill="#B9C9D7")
    draw.text((984, 674), "预热 / 无法判断：黄色", font=fonts[17], fill="#B9C9D7")
    draw.text((984, 704), "未见动作 ≠ 人员安全", font=fonts[17], fill="#B9C9D7")
    draw.text((984, 749), "研究模型 · 逐时点结果回放", font=fonts[17], fill="#93A6B7")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-dir", type=Path, default=ROOT / "runtime/checks/urfd-pose-dense-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/checks/urfd-pose-dense-v1/videos")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing a nonempty output directory")
    if not FONT.is_file() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Chinese font, ffmpeg and ffprobe are required")
    summary = read(args.check_dir / "summary.json")
    checkpoint = ROOT / "runtime/training/fall-pose-dense-v1-20260921/best.pt"
    if sha(checkpoint) != summary["checkpoint_sha256"] or summary["threshold"] != .5:
        raise ValueError("unexpected model identity or threshold")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint_sha256": sha(checkpoint), "renderer_sha256": sha(Path(__file__)), "videos": {}}
    for sequence in ("adl-10", "fall-06"):
        metadata_path = ROOT / "datasets/private/urfd-continuous-v1" / sequence / "metadata.json"
        pose_path = ROOT / "runtime/training/urfd-pose-continuous-v1" / f"{sequence}.json"
        metadata, pose = read(metadata_path), read(pose_path)
        rows = read(args.check_dir / f"{sequence}-timeline.json")["frames"]
        expected = summary["sequences"][sequence]
        if sha(metadata_path) != expected["metadata_sha256"] or sha(pose_path) != expected["raw_pose_record_sha256"]:
            raise ValueError("source identity mismatch")
        times = metadata["frame_timestamps_ms"]
        if len(rows) != len(times) or pose["timestamps_ms"] != times:
            raise ValueError("frame count or timing mismatch")
        video = args.output / f"{sequence}-new-model.mp4"
        with tempfile.TemporaryDirectory(prefix=f"urfd-demo-{sequence}-") as temporary:
            directory = Path(temporary)
            concat = ["ffconcat version 1.0"]
            for i, row in enumerate(rows):
                source = ROOT / metadata["frames"][i]
                if (sha(source) != pose["input_sha256"][i] or row["source_timestamp_ms"] != times[i]
                        or row["frame_index"] != i or row["frame_number"] != metadata["frame_numbers"][i]
                        or any(index > i for index in row["selected_indices"])):
                    raise ValueError("rendering input differs from verified timeline")
                if row["score"] is not None:
                    score = row["score"]
                    wanted = "fall_transition" if score >= .5 else "no_fall_transition"
                    if not math.isfinite(score) or not 0 <= score <= 1 or row["prediction"] != wanted:
                        raise ValueError("score and displayed prediction disagree")
                frame = draw_frame(source, sequence, row, pose, rows[:i + 1], times[-1])
                frame.save(directory / f"frame-{i:04d}.png")
                concat.extend([f"file 'frame-{i:04d}.png'", "option framerate 1000"])
                if i + 1 < len(rows):
                    concat.append(f"duration {(times[i + 1] - times[i]) / 1000:.9f}")
            (directory / "frames.ffconcat").write_text("\n".join(concat) + "\n")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", str(directory / "frames.ffconcat"), "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-enc_time_base", "1:1000",
                            "-video_track_timescale", "1000000", "-movflags", "+faststart", "-threads", "2", "-n", str(video)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time:stream=codec_name,width,height,nb_frames,duration", "-of", "json", str(video)], text=True))
        pts = [float(r["best_effort_timestamp_time"]) * 1000 for r in probe["frames"]]
        if len(pts) != len(times) or any(abs(a - b) > .01 for a, b in zip(pts, times, strict=True)):
            raise ValueError("encoded frames do not retain original source timing")
        if float(probe["streams"][0]["duration"]) * 1000 <= times[-1]:
            raise ValueError("container duration would truncate the final source frame")
        result["videos"][sequence] = {"file": str(video.resolve()), "sha256": sha(video), "frame_count": len(rows),
            "all_pts_match_source": True, "probe": probe["streams"], "timeline_sha256": sha(args.check_dir / f"{sequence}-timeline.json"),
            "playback": "original speed, no frame resampling or end hold; final display duration uses 1ms encoder time base"}
        print(f"{sequence}: verified {len(rows)} original frames and timestamps -> {video}", flush=True)
    (args.output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
