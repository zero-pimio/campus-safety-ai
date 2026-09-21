"""Render two prespecified validation examples of an offline URFD classifier."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

FPS = 25
SIZE = (960, 720)
RESULT_HOLD_SECONDS = 2
DEMO_IDS = ("adl-10", "fall-06")


def load_baseline_module():
    path = Path(__file__).with_name("train_urfd_baseline.py")
    spec = importlib.util.spec_from_file_location("urfd_baseline_visualization_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def font(size: int):
    # Pillow's bundled font avoids platform-specific font paths.
    return ImageFont.load_default(size=size)


def draw_frame(source: Image.Image, sequence: str, index: int, source_seconds: float,
               truth: str, score: float, threshold: float, prediction: str, final: bool,
               limitation: str) -> Image.Image:
    canvas = Image.new("RGB", SIZE, (17, 22, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), f"URFD | {sequence} | validation example", font=font(28), fill="white")
    draw.text((24, 55), "16 sampled frames | OFFLINE whole-clip classifier", font=font(21), fill=(189, 205, 222))
    draw.text((24, 82), limitation, font=font(16), fill=(241, 191, 108))
    display = ImageOps.contain(source, (912, 492), Image.Resampling.BILINEAR)
    canvas.paste(display, ((SIZE[0] - display.width) // 2, 104 + (492 - display.height) // 2))
    if final:
        draw.text((24, 609), f"Ground truth: {truth}  |  Model prediction: {prediction}",
                  font=font(23), fill="white")
        draw.text((24, 645), f"Full-clip fall_score: {score:.4f}  |  threshold: {threshold:.4f}",
                  font=font(23), fill=(122, 216, 210))
        draw.text((24, 682), f"Result after frame 16/16 | source +{source_seconds:.3f}s | held for 2s",
                  font=font(18), fill=(189, 205, 222))
    else:
        draw.text((24, 609), f"Ground truth (dataset): {truth}", font=font(24), fill="white")
        draw.text((24, 646), f"Input frame {index + 1:02d}/16  |  source +{source_seconds:.3f}s",
                  font=font(23), fill=(189, 205, 222))
        draw.text((24, 682), "Prediction withheld until the final sampled frame.",
                  font=font(18), fill=(189, 205, 222))
    return canvas


def encode_video(ffmpeg: str, output: Path, rendered: list[tuple[Image.Image, int]]) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s:v", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-n", str(output)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame, repetitions in rendered:
                data = frame.tobytes()
                for _ in range(repetitions):
                    process.stdin.write(data)
            process.stdin.close()
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        if returncode:
            errors.seek(0)
            raise RuntimeError("ffmpeg failed: " + errors.read().decode(errors="replace")[-2000:])


def contact_sheet(images: list[Image.Image], sample: dict, score: float,
                  threshold: float, prediction: str, truth: str, output: Path) -> None:
    sheet = Image.new("RGB", (960, 864), (17, 22, 31))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f"{sample['sequence_id']} | OFFLINE whole-clip result | 16 sampled frames",
              font=font(23), fill="white")
    draw.text((16, 47), f"Ground truth: {truth} | Model: {prediction} | score {score:.4f} | threshold {threshold:.4f}",
              font=font(19), fill=(189, 205, 222))
    first = sample["frame_timestamps_ms"][0]
    for index, image in enumerate(images):
        left, top = index % 4 * 240, 88 + index // 4 * 192
        thumbnail = ImageOps.contain(image, (228, 156), Image.Resampling.BILINEAR)
        sheet.paste(thumbnail, (left + (240 - thumbnail.width) // 2, top + (156 - thumbnail.height) // 2))
        seconds = (sample["frame_timestamps_ms"][index] - first) / 1000
        draw.text((left + 8, top + 163), f"{index + 1:02d}/16 | source +{seconds:.3f}s", font=font(15), fill="white")
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("refusing to overwrite a nonempty output directory")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required to create and verify the MP4 files")
    baseline = load_baseline_module()
    torch = baseline.torch
    torch.set_num_threads(4)
    manifest_path, checkpoint_path = args.manifest.resolve(), args.checkpoint.resolve()
    manifest_sha = baseline.sha256(manifest_path)
    checkpoint_sha = baseline.sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_sha:
        raise ValueError("checkpoint and manifest SHA256 do not match")
    if (checkpoint.get("model_name") != "fall-temporal-mobilenet-v3-small"
            or checkpoint.get("class_names") != ["non_fall", "fall"]
            or checkpoint.get("frame_count") != 16 or checkpoint.get("image_size") != 224
            or checkpoint.get("preprocess") != baseline.PREPROCESS):
        raise ValueError("checkpoint model or preprocessing protocol differs from the URFD baseline")
    threshold = float(checkpoint["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("checkpoint threshold must be finite and in [0, 1]")
    samples = baseline.validate_manifest(json.loads(manifest_path.read_text()), baseline.ROOT)
    selected = []
    for sequence, label in zip(DEMO_IDS, (0, 1), strict=True):
        matches = [sample for sample in samples if sample["sequence_id"] == sequence]
        if len(matches) != 1 or matches[0]["split"] != "val" or matches[0]["label"] != label:
            raise ValueError(f"prespecified validation example is missing or mislabeled: {sequence}")
        selected.append(matches[0])
    model = baseline.FallTemporalClassifier(pretrained=False).eval()
    model.load_state_dict(checkpoint["model_state"])
    limitation = "Research baseline | Real-time and campus performance unverified"
    diagnostic_path = checkpoint_path.parent / "validation-static-cue-diagnostic.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("checkpoint_sha256") != checkpoint_sha or diagnostic.get("split") != "val":
            raise ValueError("static-cue diagnostic must refer to this checkpoint and validation split")
        conditions = diagnostic["conditions"]
        if (conditions["first_frame_repeated"]["sample_count"] == conditions["original"]["sample_count"]
                and conditions["first_frame_repeated"]["metrics"]["accuracy"] == 1.0):
            limitation = "Research baseline | static-cue risk observed on validation data"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sample in selected:
        inputs = baseline.load_frames(sample, baseline.ROOT)
        tick = time.perf_counter()
        with torch.inference_mode():
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite demonstration logits")
            score = float(logits.softmax(dim=1)[0, 1])
        inference_ms = (time.perf_counter() - tick) * 1000
        label = baseline.predictions_at_threshold([score], threshold)[0]
        prediction, truth = checkpoint["class_names"][label], checkpoint["class_names"][sample["label"]]
        timestamps = sample["frame_timestamps_ms"]
        starts = [0]
        for timestamp in timestamps[1:]:
            starts.append(max(starts[-1] + 1, round((timestamp - timestamps[0]) / 1000 * FPS)))
        ends = starts[1:] + [starts[-1] + RESULT_HOLD_SECONDS * FPS]
        images, rendered, frame_metadata = [], [], []
        for index, relative in enumerate(sample["frames"]):
            path = baseline.ROOT / relative
            with Image.open(path) as source:
                image = source.convert("RGB")
            images.append(image)
            rendered.append((draw_frame(image, sample["sequence_id"], index,
                                        (timestamps[index] - timestamps[0]) / 1000, truth, score, threshold,
                                        prediction, index == 15, limitation), ends[index] - starts[index]))
            frame_metadata.append({"path": relative, "sha256": baseline.sha256(path),
                                   "frame_number": sample["frame_numbers"][index],
                                   "source_timestamp_ms": timestamps[index],
                                   "playback_start_seconds": starts[index] / FPS,
                                   "playback_end_seconds": ends[index] / FPS})
        video = output_dir / f"{sample['sequence_id']}-offline-demo.mp4"
        encode_video(ffmpeg, video, rendered)
        probe = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_frames,duration,r_frame_rate", "-of", "json", str(video),
        ], text=True))["streams"][0]
        if (probe["codec_name"] != "h264" or probe["pix_fmt"] != "yuv420p"
                or (probe["width"], probe["height"]) != SIZE or int(probe["nb_frames"]) != ends[-1]):
            raise ValueError("encoded video does not match the rendering contract")
        sheet = output_dir / f"{sample['sequence_id']}-contact-sheet.png"
        contact_sheet(images, sample, score, threshold, prediction, truth, sheet)
        metadata = {
            "sequence_id": sample["sequence_id"], "split": "val", "selection": "prespecified before training",
            "mode": "offline whole-sequence classifier; no per-frame predictions",
            "limitation": limitation,
            "ground_truth": truth, "prediction": prediction, "fall_score": score, "threshold": threshold,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "manifest": str(manifest_path), "manifest_sha256": manifest_sha, "preprocess": baseline.PREPROCESS,
            "inference_ms": inference_ms, "timing_scope": "single CPU forward, excluding PNG decode/preprocessing",
            "source_timestamp_origin_ms": timestamps[0], "frames": frame_metadata,
            "playback": "hold sampled frames; source timestamp deltas quantized to 25 FPS, at least one frame per sample",
            "result_hold": {"start_seconds": starts[-1] / FPS, "end_seconds": ends[-1] / FPS,
                            "duration_seconds": RESULT_HOLD_SECONDS},
            "video": str(video), "video_sha256": baseline.sha256(video), "video_probe": probe,
            "contact_sheet": str(sheet), "contact_sheet_sha256": baseline.sha256(sheet),
        }
        baseline.write_json(output_dir / f"{sample['sequence_id']}-metadata.json", metadata)
        summaries.append(metadata)
        print(json.dumps({"sequence_id": sample["sequence_id"], "video": str(video), "ground_truth": truth,
                          "prediction": prediction, "fall_score": score, "inference_ms": inference_ms}), flush=True)
    baseline.write_json(output_dir / "demonstrations.json", {"demonstrations": summaries})


if __name__ == "__main__":
    main()
