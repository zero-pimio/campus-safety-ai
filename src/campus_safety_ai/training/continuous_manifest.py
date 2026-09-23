"""Audit collected continuous videos without assigning or changing their splits."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIELDS = (
    "clip_id", "path", "split", "person_ids", "camera_id", "source_group",
    "behavior", "duration_seconds", "annotation_path", "sha256",
)
SPLITS = ("train", "val", "test")
BEHAVIORS = (
    "normal", "squatting", "bending", "sitting_down", "play_fighting",
    "lying_down_voluntarily", "fall", "fight",
)
UNKNOWN_IDS = frozenset({"", "unknown", "unspecified", "na", "n/a", "none", "null", "?"})


@dataclass(frozen=True)
class ContinuousClip:
    row: int
    clip_id: str
    path: Path
    split: str
    person_ids: tuple[str, ...]
    camera_id: str
    source_group: str
    behavior: str
    duration_seconds: float
    annotation_path: Path | None
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_continuous_manifest(
    manifest: Path, *, project_root: Path | None = None, strict: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe audit; strict additionally reads files and verifies SHA-256.

    Relative paths are anchored to project_root (cwd by default), never inferred
    from CSV placement. Missing identities never support an independence claim.
    """
    manifest = Path(manifest).resolve()
    root = (project_root or Path.cwd()).resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    clips: list[ContinuousClip] = []

    def issue(code: str, message: str, rows: list[int], *, warning: bool = False) -> None:
        target = warnings if warning else errors
        target.append({"code": code, "message": message, "rows": rows})

    def incomplete(code: str, message: str, row: int) -> None:
        issue(code, message, [row], warning=not strict)

    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(FIELDS):
                issue("invalid_header", f"CSV header must contain exactly: {', '.join(FIELDS)}", [])
            else:
                for row_number, raw in enumerate(reader, 2):
                    if None in raw or any(value is None for value in raw.values()):
                        issue("invalid_row", "Row does not have the expected number of fields", [row_number])
                        continue
                    row = {key: value.strip() for key, value in raw.items()}
                    invalid = False
                    for field in ("clip_id", "path", "source_group"):
                        if row[field].casefold() in UNKNOWN_IDS:
                            issue("missing_field", f"{field} must be explicitly known", [row_number])
                            invalid = True
                    if row["split"] not in SPLITS:
                        issue("invalid_split", "split must be train, val or test", [row_number])
                        invalid = True
                    if row["behavior"] not in BEHAVIORS:
                        issue("invalid_behavior", f"behavior must be one of {BEHAVIORS}", [row_number])
                        invalid = True
                    try:
                        duration = float(row["duration_seconds"])
                    except ValueError:
                        duration = float("nan")
                    if not math.isfinite(duration) or duration <= 0:
                        issue("invalid_duration", "duration_seconds must be finite and positive", [row_number])
                        invalid = True
                    sha = row["sha256"].lower()
                    if sha and re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                        issue("invalid_sha256", "sha256 must be 64 hexadecimal characters", [row_number])
                        invalid = True
                    if invalid:
                        continue
                    people = tuple(value.strip().casefold() for value in row["person_ids"].split(";"))
                    if any(value in UNKNOWN_IDS for value in people):
                        incomplete("unknown_person", "All visible people's stable IDs must be known", row_number)
                    people = tuple(sorted(set(people).difference(UNKNOWN_IDS)))
                    camera = row["camera_id"].casefold()
                    if camera in UNKNOWN_IDS:
                        incomplete("unknown_camera", "A stable physical camera ID must be known", row_number)
                        camera = ""
                    if not sha:
                        incomplete("missing_sha256", "Video content SHA-256 is missing", row_number)
                    annotation = row["annotation_path"]
                    if not annotation:
                        incomplete("missing_annotation", "A reviewed event annotation path is required", row_number)
                    clips.append(ContinuousClip(
                        row=row_number, clip_id=row["clip_id"], path=(root / row["path"]).resolve(),
                        split=row["split"], person_ids=people, camera_id=camera,
                        source_group=row["source_group"].casefold(), behavior=row["behavior"],
                        duration_seconds=duration,
                        annotation_path=(root / annotation).resolve() if annotation else None, sha256=sha,
                    ))
    except (OSError, UnicodeError, csv.Error) as exc:
        issue("manifest_unreadable", str(exc), [])

    if not clips:
        issue("empty_manifest", "No valid collected clips; the header-only template is not a dataset", [])

    indexes: dict[str, dict[str, list[ContinuousClip]]] = {
        name: defaultdict(list) for name in ("clip_id", "path", "person", "camera", "source_group", "sha256")
    }
    for clip in clips:
        for name, values in (
            ("clip_id", (clip.clip_id.casefold(),)), ("path", (str(clip.path),)),
            ("person", clip.person_ids), ("camera", (clip.camera_id,)),
            ("source_group", (clip.source_group,)), ("sha256", (clip.sha256,)),
        ):
            for value in values:
                if value:
                    indexes[name][value].append(clip)

    for name, index in indexes.items():
        for key, members in index.items():
            rows = [clip.row for clip in members]
            if name == "clip_id" and len(members) > 1:
                issue("duplicate_clip_id", f"Duplicate clip ID: {key}", rows)
            if len({clip.split for clip in members}) > 1:
                issue(f"{name}_cross_split", f"{name} occurs across splits: {key}", rows)
            if name in {"path", "sha256"} and len({clip.source_group for clip in members}) > 1:
                issue("duplicate_content_groups", f"Same {name} assigned to multiple source groups: {key}", rows)

    verified_hashes: dict[Path, str] = {}
    if strict:
        for clip in clips:
            if not clip.path.is_file():
                issue("missing_video", f"Video does not exist as a file: {clip.path}", [clip.row])
            else:
                try:
                    if clip.path.stat().st_size == 0:
                        issue("empty_video", f"Video is empty: {clip.path}", [clip.row])
                    else:
                        if clip.path not in verified_hashes:
                            verified_hashes[clip.path] = _sha256(clip.path)
                        if clip.sha256 and verified_hashes[clip.path] != clip.sha256:
                            issue("sha256_mismatch", f"Video SHA-256 mismatch: {clip.path}", [clip.row])
                except OSError as exc:
                    issue("video_unreadable", str(exc), [clip.row])
            if clip.annotation_path is not None and not clip.annotation_path.is_file():
                issue("missing_annotation_file", f"Annotation does not exist: {clip.annotation_path}", [clip.row])

        # Recompute duplicate detection independently of the declared hashes.
        actual_hash_groups: dict[str, list[ContinuousClip]] = defaultdict(list)
        for clip in clips:
            actual = verified_hashes.get(clip.path)
            if actual:
                actual_hash_groups[actual].append(clip)
        for actual, members in actual_hash_groups.items():
            if len({clip.split for clip in members}) > 1:
                issue("verified_sha256_cross_split", f"Identical video bytes occur across splits: {actual}",
                      [clip.row for clip in members])
            if len({clip.source_group for clip in members}) > 1:
                issue("verified_duplicate_content_groups", "Identical video bytes have multiple source groups",
                      [clip.row for clip in members])

    def summary(members: list[ContinuousClip]) -> dict[str, Any]:
        return {
            "clips": len(members), "independent_source_groups": len({clip.source_group for clip in members}),
            "unique_video_paths": len({clip.path for clip in members}),
            "listed_clip_duration_seconds": sum(clip.duration_seconds for clip in members),
            "behaviors": {
                behavior: {
                    "clips": sum(clip.behavior == behavior for clip in members),
                    "independent_source_groups": len({
                        clip.source_group for clip in members if clip.behavior == behavior
                    }),
                }
                for behavior in BEHAVIORS
            },
            "missing_behaviors": [behavior for behavior in BEHAVIORS
                                  if not any(clip.behavior == behavior for clip in members)],
        }

    by_split = {split: summary([clip for clip in clips if clip.split == split]) for split in SPLITS}
    missing_splits = [split for split in SPLITS if not by_split[split]["clips"]]
    if missing_splits:
        issue("missing_splits", f"No clips in splits: {', '.join(missing_splits)}", [], warning=not strict)
    for split, counts in by_split.items():
        if counts["missing_behaviors"]:
            issue("behavior_coverage_gap", f"{split} lacks: {', '.join(counts['missing_behaviors'])}", [], warning=True)

    identity_complete = bool(clips) and not any(
        item["code"] in {"unknown_person", "unknown_camera"} for item in errors + warnings
    )
    return {
        "schema_version": "1.0", "manifest": str(manifest), "project_root": str(root), "strict": strict,
        "passed": not errors, "status": "failed" if errors else "strict_passed" if strict else "metadata_only",
        "file_hashes_verified": len(verified_hashes), "identities_complete": identity_complete,
        "eligible_for_person_camera_disjoint_evaluation": strict and not errors and identity_complete,
        "totals": summary(clips), "by_split": by_split,
        "errors": errors, "warnings": warnings,
        "limitations": [
            "Identity and source-group correctness depend on reviewed provenance; IDs cannot be inferred from names.",
            "SHA-256 detects byte-identical files, not re-encoded, cropped or overlapping copies.",
            "Source-group counts describe declared groups, not proven statistical independence.",
            "Listed durations may overlap and are not exposure hours for false-alarm metrics.",
            "This audit does not validate annotation semantics, media duration, licensing or model accuracy.",
        ],
    }
