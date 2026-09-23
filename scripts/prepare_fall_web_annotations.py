#!/usr/bin/env python3
"""Rebuild auditable fall-web-v1 annotations from pinned local downloads.

No inference, downloading, training, or original inventory mutation is performed.
All frame times are decoded PTS or original PNG filename timestamps, never FPS estimates.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BATCH = Path("datasets/private/web-hard-negatives-20260923")
OUTPUT = Path("runtime/annotations/fall-web-v1")
MANIFEST = Path("datasets/manifests/fall-web-v1.json")
SPLITS = {"gmdcsa24:subject-1": "train", "gmdcsa24:subject-2": "train",
          "gmdcsa24:subject-3": "val", "gmdcsa24:subject-4": "test"}


# These decoded-frame indices were visually reviewed by the assistant on 2026-09-23.
# Tuple: earliest onset, certain descent, earliest landing, latest landing, static posture.
# Wide bounds deliberately retain uncertainty in slowly staged leaning motions.
FALL_REVIEW = {
    "gmdcsa24-subject-1-fall-03": (29, 73, 93, 101, 106),
    "gmdcsa24-subject-1-fall-08": (20, 26, 54, 60, 64),
    "gmdcsa24-subject-2-fall-01": (36, 54, 88, 100, 107),
    "gmdcsa24-subject-2-fall-04": (98, 104, 125, 129, 185),
    "gmdcsa24-subject-3-fall-03": (39, 49, 65, 71, 89),
    "gmdcsa24-subject-3-fall-04": (30, 40, 56, 62, 78),
    "gmdcsa24-subject-4-fall-03": (47, 54, 68, 76, None),
    "up-fall-Subject1Activity1Trial1Camera2": (63, 68, 82, 87, 92),
}
REFINE_RANGES = {
    "gmdcsa24-subject-1-fall-03": (1.8, 3.5),
    "gmdcsa24-subject-1-fall-08": (0.5, 2.2),
    "gmdcsa24-subject-2-fall-01": (1.0, 3.6),
    "gmdcsa24-subject-2-fall-04": (3.0, 4.8),
    "gmdcsa24-subject-3-fall-03": (1.1, 2.8),
    "gmdcsa24-subject-3-fall-04": (0.8, 2.6),
    "gmdcsa24-subject-4-fall-03": (1.0, 3.5),
}
REVIEW_NOTES = {
    "gmdcsa24-subject-1-fall-03": "Slow preparatory lean makes onset ambiguous; clear chair departure to floor descent supplies the positive core.",
    "gmdcsa24-subject-1-fall-08": "Walking, knee flexion and forward/side descent; prolonged floor lying excluded from fall-motion positives.",
    "gmdcsa24-subject-2-fall-01": "Slow staged lateral lean; broad onset bracket retained. Hands touch before torso/pelvis landing.",
    "gmdcsa24-subject-2-fall-04": "Backward descent lands seated; subsequent seated-to-supine movement is post-fall transition, not recovery or another fall.",
    "gmdcsa24-subject-3-fall-03": "Backward descent to pelvis support, then reclining and leg settling; transition excluded from static negatives.",
    "gmdcsa24-subject-3-fall-04": "Walking followed by backward descent; rolling after landing excluded from static negatives.",
    "gmdcsa24-subject-4-fall-03": "Forward descent into partly cropped foreground; broad landing bracket. Arm/posture movement and cropping leave tail excluded from static negatives.",
    "up-fall-Subject1Activity1Trial1Camera2": "Designated actor falls onto mat while an unidentified helper remains visible; diagnostic only, no person-disjoint or model-selection claim.",
}
ADL_REVIEW = {
    "gmdcsa24-subject-1-adl-01": ("voluntary_lying_down", "Seated on bed, deliberately reclines onto side."),
    "gmdcsa24-subject-1-adl-11": ("sitting_down", "Walks to chair, sits, reads."),
    "gmdcsa24-subject-1-adl-15": ("bending_and_sitting", "Bends to retrieve object, stands, sits on chair."),
    "gmdcsa24-subject-1-adl-16": ("seated_bending", "Leans to floor while remaining seated, returns upright."),
    "gmdcsa24-subject-2-adl-03": ("voluntary_lying_down", "Bed/pillow activity followed by deliberate sitting and reclining."),
    "gmdcsa24-subject-2-adl-08": ("bending_exercise", "Repeated standing bend/stretch to floor and return upright."),
    "gmdcsa24-subject-2-adl-09": ("sitting_down", "Stands from bed, walks, deliberately sits on floor."),
    "gmdcsa24-subject-2-adl-12": ("voluntary_lying_down", "Climbs onto bed and deliberately lies prone."),
    "gmdcsa24-subject-3-adl-05": ("bending_exercise", "Repeated bend and extend exercise."),
    "gmdcsa24-subject-3-adl-06": ("bending_and_sitting", "Bends during exercise then deliberately sits on floor."),
    "gmdcsa24-subject-3-adl-07": ("squat_like_exercise", "Repeated deep knee flexion and rising visible; source calls it sitting/standing exercise."),
    "gmdcsa24-subject-3-adl-08": ("voluntary_lying_down", "Seated on floor then deliberately reclines."),
    "gmdcsa24-subject-4-adl-05": ("push_up_exercise", "Walks in, lowers to push-up position, repeated exercise; foreground partly cropped."),
    "gmdcsa24-subject-4-adl-10": ("voluntary_lying_down", "Walks to bed, sits then reclines."),
    "gmdcsa24-subject-4-adl-15": ("voluntary_lying_down", "Sitting on bed then deliberately lying on bed."),
    "gmdcsa24-subject-4-adl-17": ("bending", "Walks in, bends to retrieve book, stands, puts book on bed."),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_times(times: list[float]) -> None:
    if not times or not all(math.isfinite(x) for x in times):
        raise ValueError("frame timestamps must be nonempty and finite")
    if abs(times[0]) > 1e-6 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("frame timestamps must start at zero and strictly increase")


def timing_from_probe(data: dict[str, Any]) -> dict[str, Any]:
    frames = data["frames"]
    original = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    times = [round(t - original[0], 6) for t in original]
    validate_times(times)
    last_duration = float(frames[-1].get("duration_time", frames[-1].get("pkt_duration_time", 0)))
    if not math.isfinite(last_duration) or last_duration <= 0:
        raise ValueError("last frame needs an explicit positive decoded duration")
    durations = [round(b - a, 6) for a, b in zip(times, times[1:], strict=False)] + [last_duration]
    return {"time_source": "ffprobe decoded best_effort_timestamp_time; first PTS subtracted",
            "first_pts_s": original[0], "frame_count": len(times), "frame_timestamps_s": times,
            "frame_durations_s": durations, "duration_s": round(times[-1] + last_duration, 6),
            "frame_index_base": 0, "streams": data.get("streams", [])}


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time,pkt_duration_time,duration_time",
        "-show_entries", "stream=time_base,avg_frame_rate,start_time,duration,width,height",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return timing_from_probe(json.loads(result.stdout))


def original_png_timing(data: dict[str, Any]) -> dict[str, Any]:
    frames = data["original_frames"]
    values = [datetime.fromisoformat(frame["timestamp"]) for frame in frames]
    if any(datetime.fromisoformat(Path(frame["name"]).stem.replace("_", ":")) != value
           for frame, value in zip(frames, values, strict=True)):
        raise ValueError("PNG filename and declared timestamp disagree")
    times = [round((t - values[0]).total_seconds(), 6) for t in values]
    validate_times(times)
    return {"time_source": "original PNG filename timestamps; first timestamp subtracted",
            "frame_count": len(times), "frame_index_base": 0, "frame_timestamps_s": times,
            "frame_names": [f["name"] for f in frames], "first_original_timestamp": frames[0]["timestamp"],
            "original_timezone": "not_stated", "duration_s": times[-1],
            "duration_basis": "observed first-to-last frame span; final frame exposure unknown"}


def nearest_indices(times: list[float], start: float, end: float, step: float) -> list[int]:
    if step <= 0 or start < 0 or end < start:
        raise ValueError("invalid contact sheet time range")
    selected = []
    for n in range(int(math.floor((end - start) / step)) + 1):
        t = start + n * step
        index = min(bisect.bisect_left(times, t), len(times) - 1)
        if index and abs(times[index - 1] - t) < abs(times[index] - t):
            index -= 1
        if not selected or index != selected[-1]:
            selected.append(index)
    return selected


def contact_sheet(video: Path, timing: dict[str, Any], output: Path, *, step: float,
                  start: float = 0, end: float | None = None, title: str = "") -> dict[str, Any]:
    import cv2
    from PIL import Image, ImageDraw
    times = timing["frame_timestamps_s"]
    indices = nearest_indices(times, start, times[-1] if end is None else end, step)
    selected = set(indices)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    frames = {}
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded in selected:
            frames[decoded] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decoded += 1
    cap.release()
    if decoded != len(times) or set(frames) != selected:
        raise ValueError(f"decode/PTS count mismatch {video}: {decoded} != {len(times)}")
    width, height = 240, 158
    canvas = Image.new("RGB", (6 * width, 28 + math.ceil(len(indices) / 6) * height), "#f4f4f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for n, index in enumerate(indices):
        x, y = n % 6 * width, 28 + n // 6 * height
        im = frames[index]
        im.thumbnail((width - 4, height - 24))
        canvas.paste(im, (x + (width - im.width) // 2, y))
        draw.text((x + 4, y + height - 20), f"frame {index:03}  t={times[index]:.3f}s", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    record = {"path": str(output.relative_to(ROOT)), "sha256": sha256(output),
              "frame_indices": indices, "timestamps_s": [times[i] for i in indices],
              "sample_period_s": step, "decoded_frames_verified": decoded}
    write_json(output.with_suffix(".json"), record)
    return record


def generate_evidence() -> list[dict[str, Any]]:
    inventory = json.loads((ROOT / BATCH / "gmdcsa24/inventory.json").read_text())
    rows = []
    for source in inventory["videos"]:
        sample_id = "gmdcsa24-" + source["original_path"].lower().replace(" ", "-").replace("/", "-").removesuffix(".mp4")
        video = ROOT / BATCH / "gmdcsa24" / source["local_path"]
        if sha256(video) != source["sha256"]:
            raise ValueError(f"source SHA mismatch: {video}")
        timing = probe_video(video)
        folder = ROOT / OUTPUT / sample_id
        write_json(folder / "timing.json", timing)
        evidence = contact_sheet(video, timing, folder / "contact.jpg",
                                 step=0.2 if source["source_category"] == "Fall" else 0.6,
                                 title=source["original_path"] + " | relative decoded PTS")
        evidence_list = [evidence]
        if sample_id in REFINE_RANGES:
            start, end = REFINE_RANGES[sample_id]
            evidence_list.append(contact_sheet(video, timing, folder / "contact-refine.jpg", step=0.0666667,
                                               start=start, end=end, title=source["original_path"] + " | refine relative PTS"))
        rows.append({"sample_id": sample_id, "source": source, "video_path": str(video.relative_to(ROOT)),
                     "timing": timing, "evidence": evidence_list})
    up = json.loads((ROOT / BATCH / "up-fall/manifest.json").read_text())
    for source in up["samples"]:
        sample_id = "up-fall-" + source["sample_id"]
        video = ROOT / source["derived_video"]
        if sha256(video) != source["derived_video_sha256"]:
            raise ValueError(f"source SHA mismatch: {video}")
        sidecar = ROOT / BATCH / "up-fall" / (source["sample_id"] + ".frames.json")
        timing = original_png_timing(json.loads(sidecar.read_text()))
        folder = ROOT / OUTPUT / sample_id
        write_json(folder / "timing.json", timing)
        write_json(folder / "preview-timing.json", probe_video(video))
        evidence = contact_sheet(video, timing, folder / "contact.jpg", step=0.25,
                                 title=source["sample_id"] + " | original PNG times via ordered preview frames")
        rows.append({"sample_id": sample_id, "source": source, "video_path": str(video.relative_to(ROOT)),
                     "timing": timing, "evidence": [evidence]})
    write_json(ROOT / OUTPUT / "evidence-index.json", rows)
    return rows


def interval(start: float, end: float, label: str, target: int | None, basis: str) -> dict[str, Any]:
    return {"start_s": start, "end_s": end, "label": label, "training_label": target, "basis": basis}


def fall_intervals(sample_id: str, timing: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = FALL_REVIEW[sample_id]
    t = timing["frame_timestamps_s"]
    onset_lo, onset_hi, landing_lo, landing_hi = [t[index] for index in indices[:4]]
    static = t[indices[4]] if indices[4] is not None else timing["duration_s"]
    intervals = [
        interval(0, onset_lo, "non_fall", 0, "pre-event visual review"),
        interval(onset_lo, onset_hi, "uncertain", None, "fall onset bracket; not exact ground truth"),
        interval(onset_hi, landing_lo, "fall_motion", 1, "visually confirmed descent core"),
        interval(landing_lo, landing_hi, "uncertain", None, "landing bracket; early hand contact is not landing"),
        interval(landing_hi, static, "post_fall_transition", None, "repositioning after landing; no recovery claim"),
        interval(static, timing["duration_s"], "post_fall", 0,
                 "static supported posture is negative for fall MOTION only; not recovered"),
    ]
    intervals = [row for row in intervals if row["end_s"] > row["start_s"]]
    event = {"event_id": sample_id + "-fall-1", "start_s": onset_lo, "end_s": landing_hi,
             "onset_range_s": [onset_lo, onset_hi], "landing_range_s": [landing_lo, landing_hi],
             "confirmed_motion_range_s": [onset_hi, landing_lo],
             "boundary_frame_indices": list(indices), "boundary_precision": "visual bracket, not exact onset",
             "recovery_observed": False, "recovery_time_s": None,
             "recovery_status": "not visible in sampled evidence; right-censored recording",
             "event_scope": "fall descent through landing, not persistent prone state"}
    return intervals, [event]


def verify_original_csv(source: dict[str, Any]) -> tuple[str, str]:
    path = ROOT / BATCH / "gmdcsa24" / source["source_annotation_path"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["File Name"] == Path(source["original_path"]).name]
    if len(rows) != 1 or rows[0] != source["source_annotation"]:
        raise ValueError(f"original CSV no longer matches inventory: {source['original_path']}")
    return str(path.relative_to(ROOT)), sha256(path)


def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for row in rows:
        source, sample_id, timing = row["source"], row["sample_id"], row["timing"]
        gmd = sample_id.startswith("gmdcsa24-")
        digest = source["sha256"] if gmd else source["derived_video_sha256"]
        if sha256(ROOT / row["video_path"]) != digest:
            raise ValueError(f"source SHA mismatch: {sample_id}")
        validate_times(timing["frame_timestamps_s"])
        timing_path = ROOT / OUTPUT / sample_id / "timing.json"
        if json.loads(timing_path.read_text()) != timing:
            raise ValueError(f"timing sidecar and cached evidence disagree: {sample_id}")
        if gmd and probe_video(ROOT / row["video_path"]) != timing:
            raise ValueError(f"decoded timestamps changed: {sample_id}")
        fall = sample_id in FALL_REVIEW
        if fall:
            intervals, events = fall_intervals(sample_id, timing)
            note = REVIEW_NOTES[sample_id]
            behavior = "fall"
        else:
            intervals = [interval(0, timing["duration_s"], "non_fall", 0,
                                  "official ADL/activity coarse negative with visual contact-sheet spotcheck")]
            events = []
            behavior, note = ADL_REVIEW.get(sample_id, ("bending", "Bends to pick up object, returns upright."))
        if not gmd:
            for item in intervals:
                item["training_label"] = None
        evidence = row["evidence"][:]
        refine = ROOT / OUTPUT / sample_id / "contact-refine.json"
        if refine.exists() and not any("contact-refine" in e["path"] for e in evidence):
            evidence.append(json.loads(refine.read_text()))
        for artifact in evidence:
            if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"evidence SHA mismatch: {artifact['path']}")
        sample = {
            "sample_id": sample_id, "dataset": "GMDCSA24" if gmd else "UP-Fall",
            "split": SPLITS[source["subject_id"]] if gmd else "external_diagnostic",
            "subject_id": source["subject_id"] if gmd else source["designated_actor_id"],
            "person_ids": [source["subject_id"]] if gmd else [source["designated_actor_id"]],
            "additional_visible_persons_unidentified": False if gmd else source["additional_visible_persons_unidentified"],
            "person_identity_basis": "source-declared subject only; visual identity not independently authenticated",
            "camera_id": source["camera_id"], "source_group": source["source_group"],
            "video_path": row["video_path"], "sha256": digest,
            "original_path": source["original_path"] if gmd else source["sample_id"] + ".zip",
            "source_url": source["url"] if gmd else source["official_download_url"],
            "source_category": source["source_category"] if gmd else source["official_activity_label"],
            "source_annotation": source["source_annotation"] if gmd else source["official_activity_label"],
            "timing": timing, "interval_semantics": "half-open [start_s, end_s)",
            "intervals": intervals, "events": events, "behavior": behavior,
            "annotation_review": {"reviewer": "assistant visual review", "reviewed_date": "2026-09-23",
                                  "independent_human_review": False,
                                  "method": "official label plus dense/spotcheck contact sheets; no model predictions",
                                  "note": note, "precision": "bounded event times" if fall else "coarse negative"},
            "evidence": evidence,
            "timing_sidecar": str(OUTPUT / sample_id / "timing.json"),
            "timing_sidecar_sha256": sha256(timing_path),
            "strict_person_camera_admission_passed": False,
        }
        if gmd:
            path, csv_hash = verify_original_csv(source)
            sample.update({"source_annotation_path": path, "source_annotation_sha256": csv_hash,
                           "license": "MIT per source repository; video rights chain not separately verified"})
        else:
            archive = ROOT / BATCH / "up-fall" / (source["sample_id"] + ".zip")
            if sha256(archive) != source["archive_sha256"]:
                raise ValueError(f"original archive SHA mismatch: {sample_id}")
            frame_sidecar = ROOT / BATCH / "up-fall" / (source["sample_id"] + ".frames.json")
            if original_png_timing(json.loads(frame_sidecar.read_text())) != timing:
                raise ValueError(f"original PNG timing changed: {sample_id}")
            with zipfile.ZipFile(archive) as zipped:
                names = sorted(name for name in zipped.namelist() if name.lower().endswith(".png"))
            if names != timing["frame_names"]:
                raise ValueError(f"original PNG names do not match archive: {sample_id}")
            sample.update({"archive_path": str(archive.relative_to(ROOT)),
                           "archive_sha256": source["archive_sha256"],
                           "data_license": source["data_license"],
                           "research_use_basis": source["research_use_basis"],
                           "preview_timing_is_approximate": True,
                           "original_frames_sidecar_sha256": sha256(frame_sidecar),
                           "original_frames_sidecar": str(BATCH / "up-fall" / (source["sample_id"] + ".frames.json"))})
        validate_intervals(sample)
        write_json(ROOT / OUTPUT / sample_id / "annotation.json", sample)
        samples.append(sample)
    validate_splits(samples)
    return {"schema_version": "1.0", "dataset": "fall-web-v1", "frozen_date": "2026-09-23",
            "status": "frozen research annotation and subject-group split; not deployment acceptance",
            "split_policy": {"GMDCSA24": SPLITS, "UP-Fall": "external_diagnostic only; never train or tune",
                             "basis": "source-declared subject plus conservative same-subject source grouping",
                             "cross_camera_independence_verified": False,
                             "test_scores_seen_during_annotation": False},
            "label_contract": {"task": "fall motion, not lying-person detection",
                               "uncertain_or_post_fall_transition": "exclude unless separately frozen positive-core overlap policy applies",
                               "post_fall_zero": "static motion negative only; never means recovery",
                               "coarse_negative": "official ADL label and visual spotcheck; not frame-exhaustive human annotation",
                               "event_envelope": "earliest possible onset through latest landing; report boundary uncertainty",
                               "external_diagnostic": "all training_label values null; never used for selection"},
            "limitations": ["Only four source-declared GMD subjects and seven GMD falls; not campus validation.",
                            "GMD physical camera identities are unknown. Subject2 and Subject3 share a visually similar room/view; camera leakage is unresolved.",
                            "No new-camera generalization claim or strict person/camera audit pass.",
                            "Original recording independence is unresolved; source groups conservatively equal subjects.",
                            "No observed fall-to-standing recovery; source end must not be labeled recovery.",
                            "Contact-sheet annotation was done by the assistant, not independently double reviewed by a human.",
                            "Short staged clips are not representative continuous exposure for deployment false-alarms/hour.",
                            "UP-Fall has one designated subject and an unidentified visible helper; research diagnostic only."],
            "source_manifests": [{"path": str(BATCH / "gmdcsa24/inventory.json"),
                                  "sha256": sha256(ROOT / BATCH / "gmdcsa24/inventory.json")},
                                 {"path": str(BATCH / "up-fall/manifest.json"),
                                  "sha256": sha256(ROOT / BATCH / "up-fall/manifest.json")}],
            "samples": samples}


def validate_intervals(sample: dict[str, Any]) -> None:
    cursor = 0.0
    for item in sample["intervals"]:
        start, end = item["start_s"], item["end_s"]
        if not all(math.isfinite(t) for t in (start, end)) or abs(start - cursor) > 1e-6 or end <= start:
            raise ValueError("intervals must be finite, contiguous and strictly positive")
        if item["label"] in ("uncertain", "post_fall_transition") and item["training_label"] is not None:
            raise ValueError("uncertain movement cannot silently become positive or negative")
        cursor = end
    if abs(cursor - sample["timing"]["duration_s"]) > 1e-6:
        raise ValueError("intervals must cover exactly the observed duration")
    for event in sample["events"]:
        a, b = event["onset_range_s"]
        c, d = event["landing_range_s"]
        if not (0 <= a <= b < c <= d <= cursor):
            raise ValueError("fall temporal brackets must be ordered")
        if event["recovery_observed"] or event["recovery_time_s"] is not None:
            raise ValueError("this batch contains no annotated recovery")


def validate_splits(samples: list[dict[str, Any]]) -> None:
    for key in ("subject_id", "source_group", "sha256", "video_path"):
        assignments: dict[str, str] = {}
        for sample in samples:
            value, split = sample[key], sample["split"]
            if value in assignments and assignments[value] != split:
                raise ValueError(f"{key} crosses splits: {value}")
            assignments[value] = split
    for sample in samples:
        if sample["dataset"] == "GMDCSA24" and sample["camera_id"] is not None:
            raise ValueError("GMD camera identities are unknown; do not invent IDs")
        if sample["dataset"] == "UP-Fall" and sample["split"] != "external_diagnostic":
            raise ValueError("UP-Fall must remain external diagnostic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--reuse-evidence", action="store_true", help="Use existing verified contact sheets and decoded timing")
    args = parser.parse_args()
    rows = (json.loads((ROOT / OUTPUT / "evidence-index.json").read_text())
            if args.reuse_evidence else generate_evidence())
    if not args.evidence_only:
        manifest = build_manifest(rows)
        write_json(ROOT / MANIFEST, manifest)
        write_json(ROOT / OUTPUT / "freeze.json", {"manifest_path": str(MANIFEST),
                                                   "manifest_sha256": sha256(ROOT / MANIFEST),
                                                   "sample_count": len(rows),
                                                   "model_inference_used_for_review": False})
    print(json.dumps({"samples": len(rows), "evidence_directory": str(OUTPUT),
                      "manifest": None if args.evidence_only else str(MANIFEST)}))


if __name__ == "__main__":
    main()
