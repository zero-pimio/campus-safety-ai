"""Fetch complete synchronized RGB frames for eight fixed URFD train sequences."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import zipfile
from pathlib import Path, PurePosixPath

import prepare_urfd as original
from prepare_urfd_continuous import ChunkedArchive

DATA_PATH = Path("datasets/private/urfd-dense-train-v1")
SEQUENCES = ("adl-01", "adl-02", "adl-04", "adl-06", "fall-01", "fall-04", "fall-05", "fall-09")


def prepare_sequence(root: Path, sample: dict, manifest_sha: str) -> dict:
    sequence = sample["sequence_id"]
    if sequence not in SEQUENCES or sample["split"] != "train":
        raise ValueError("only the eight preselected original train sequences may be prepared")
    directory = root / DATA_PATH / sequence
    if directory.is_symlink() or (directory / "archive-parts").is_symlink():
        raise ValueError("refusing symlink output directory")
    directory.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    with original.HttpClient() as client:
        identity = client.head(sample["source_url"])
        if (identity["content_length"] != sample["archive_size"]
                or identity["etag"] != sample["archive_etag"]
                or identity["last_modified"] != sample["archive_last_modified"]):
            raise ValueError("source archive identity differs from the original manifest")
        marker = directory / "archive.json"
        if not marker.exists() and any(directory.iterdir()):
            raise ValueError("unrecognized nonempty output directory")
        original.immutable_write(marker, original.json_bytes(identity))
        print(f"source {sequence} archive_bytes={identity['content_length']}", flush=True)
        sync_data = original.cached_small(client, sample["sync_url"], directory / "sync.csv")
        if original.sha256(sync_data) != sample["sync_sha256"]:
            raise ValueError("official sync differs from the original manifest")
        times = original.synchronization(sync_data)
        remote = ChunkedArchive(client, identity, directory / "archive-parts")
        with zipfile.ZipFile(remote) as archive:
            members = original.png_members(archive, sequence)
            if len(members) != sample["member_count"]:
                raise ValueError("archive member count differs from the original manifest")
            retained = [(number, info) for number, info in members if number in times]
            excluded = [number for number, _ in members if number not in times]
            if len(retained) < 2:
                raise ValueError("at least two synchronized RGB frames are required")
            timestamps = [times[number] for number, _ in retained]
            if (any(not math.isfinite(value) or value < 0 for value in timestamps)
                    or any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False))):
                raise ValueError("official timestamps must be finite, nonnegative, and increasing")
            allowed = {PurePosixPath(info.filename).name for _, info in retained} | {
                "archive.json", "archive-parts", "sync.csv", "sync.csv.json", "metadata.json",
            }
            unknown = [path.name for path in directory.iterdir()
                       if path.name not in allowed and not path.name.startswith(".urfd-write-")]
            if unknown:
                raise ValueError(f"unrecognized files: {unknown}")
            sparse_hashes = {member["frame_number"]: member["sha256"] for member in sample["members"]}
            provenance, dimensions = [], set()
            for index, (number, info) in enumerate(retained):
                path = directory / PurePosixPath(info.filename).name
                if path.is_symlink():
                    raise ValueError("refusing symlink PNG")
                data = path.read_bytes() if path.exists() else archive.read(info)
                dimensions.add(original.validate_png(data, info))
                digest = original.sha256(data)
                if number in sparse_hashes and digest != sparse_hashes[number]:
                    raise ValueError("RGB frame differs from the original sparse dataset")
                original.immutable_write(path, data)
                provenance.append({
                    "frame_number": number, "timestamp_ms": times[number],
                    "path": path.relative_to(root).as_posix(), "member_name": info.filename,
                    "crc32": f"{info.CRC:08x}", "sha256": digest, "size": len(data),
                })
                if (index + 1) % 50 == 0 or index + 1 == len(retained):
                    print(f"frames {sequence} {index + 1}/{len(retained)} "
                          f"elapsed={time.monotonic() - tick:.1f}s", flush=True)
            if len(dimensions) != 1:
                raise ValueError("dimensions changed within sequence")
            archive_hash = remote.archive_sha256()
        width, height = dimensions.pop()
        if (width, height) != (sample["width"], sample["height"]):
            raise ValueError("dimensions differ from the original manifest")
        result = {
            "schema_version": "1.0", "dataset": "urfd-dense-train-v1",
            "sequence_id": sequence, "split": "train", "label": sample["label"],
            "group_id": sample["group_id"], "original_manifest_sha256": manifest_sha,
            "selection_basis": "eight fixed original train sequence IDs chosen before dense retrieval",
            "source_url": identity["url"], "archive_size": identity["content_length"],
            "archive_etag": identity["etag"], "archive_last_modified": identity["last_modified"],
            "archive_sha256": archive_hash, "member_count": len(members), "frame_count": len(provenance),
            "excluded_unsynchronized_frame_numbers": excluded, "width": width, "height": height,
            "frames": [item["path"] for item in provenance],
            "frame_numbers": [item["frame_number"] for item in provenance],
            "frame_timestamps_ms": timestamps, "members": provenance,
            "sync_url": sample["sync_url"], "sync_sha256": original.sha256(sync_data),
            "license": "CC-BY-NC-SA-4.0", "source_page": original.BASE_URL + "uf.html",
            "timestamp_policy": "Retain every RGB frame with an official synchronization row; no FPS guessing",
            "inference_input_note": "Use original PNGs and frame_timestamps_ms; no MP4 is generated.",
            "label_basis": sample["label_basis"],
        }
        # The completion marker is published only after every frame and archive is verified.
        original.immutable_write(directory / "metadata.json", original.json_bytes(result))
        print(f"complete {sequence} frames={len(provenance)} elapsed={time.monotonic() - tick:.1f}s "
              f"downloaded_bytes_this_run={remote.downloaded_bytes}", flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    import cv2

    cv2.setNumThreads(1)
    root = args.project_root.resolve()
    original_manifest = root / original.MANIFEST_PATH
    manifest_data = original_manifest.read_bytes()
    manifest_sha = original.sha256(manifest_data)
    samples = json.loads(manifest_data)["samples"]
    selected = {}
    for sample in samples:
        if sample["sequence_id"] in SEQUENCES:
            if sample["sequence_id"] in selected or sample["split"] != "train":
                raise ValueError("duplicate or non-train sequence in the original manifest")
            planned = next(row for row in original.split_plan() if row["sequence_id"] == sample["sequence_id"])
            if any(sample[key] != planned[key] for key in ("split", "label", "group_id")):
                raise ValueError("original manifest disagrees with the fixed split plan")
            selected[sample["sequence_id"]] = sample
    if set(selected) != set(SEQUENCES):
        raise ValueError("fixed train sequences are missing from the original manifest")
    directory = root / DATA_PATH
    if directory.is_symlink():
        raise ValueError("refusing symlink dataset directory")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "dataset.json"
    if not marker.exists() and any(directory.iterdir()):
        raise ValueError("refusing unrecognized dataset directory")
    original.immutable_write(marker, original.json_bytes({
        "schema_version": "1.0", "dataset": "urfd-dense-train-v1", "sequences": list(SEQUENCES),
        "original_manifest_sha256": manifest_sha, "license": "CC-BY-NC-SA-4.0",
        "source_page": original.BASE_URL + "uf.html", "purpose": "complete synchronized original train RGB frames",
    }))
    page = root / "datasets/private/urfd-cam0/license-page.html"
    original.immutable_write(directory / "license-page.html", page.read_bytes())
    original.immutable_write(directory / "license-page.html.json", original.json_bytes({
        "url": original.BASE_URL + "uf.html", "sha256": original.sha256(page.read_bytes()),
        "copied_from": page.relative_to(root).as_posix(), "license": "CC-BY-NC-SA-4.0",
    }))
    print(f"plan archives_bytes={sum(row['archive_size'] for row in selected.values())} "
          f"rgb_frames={sum(row['member_count'] for row in selected.values())} workers={args.workers}", flush=True)

    def prepare(sequence):
        for attempt in range(3):
            try:
                return prepare_sequence(root, selected[sequence], manifest_sha)
            except Exception as error:
                print(f"retry {sequence} {attempt + 1}/3: {error}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare, SEQUENCES))
    if original.sha256(original_manifest.read_bytes()) != manifest_sha:
        raise ValueError("original manifest changed during retrieval")
    original.immutable_write(directory / "manifest.json", original.json_bytes({
        "schema_version": "1.0", "dataset": "urfd-dense-train-v1", "samples": results,
        "original_manifest_sha256": manifest_sha,
    }))


if __name__ == "__main__":
    main()
