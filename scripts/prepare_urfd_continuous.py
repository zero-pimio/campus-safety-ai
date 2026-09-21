"""Prepare two fixed validation sequences as complete synchronized RGB streams."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import prepare_urfd as original

DATA_PATH = Path("datasets/private/urfd-continuous-v1")
SEQUENCES = ("adl-10", "fall-06")
CHUNK_SIZE = 4 * 1024 * 1024


class ChunkedArchive(io.RawIOBase):
    """A seekable ZIP backed by immutable bounded HTTP Range chunks."""

    def __init__(self, client, identity, directory):
        self.client, self.identity, self.directory = client, identity, directory
        self.size = identity["content_length"]
        self.position = 0
        self.cached_index, self.cached_data = None, None
        self.downloaded_bytes = 0
        directory.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("invalid archive seek")
        return self.position

    def chunk(self, index):
        if self.cached_index == index:
            return self.cached_data
        start = index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, self.size) - 1
        if start > end:
            raise ValueError("invalid chunk index")
        path = self.directory / f"{index:06d}.bin"
        metadata_path = path.with_suffix(".json")
        if path.is_symlink() or metadata_path.is_symlink():
            raise ValueError("refusing symlink archive cache")
        if path.exists():
            if not metadata_path.is_file():
                raise ValueError("unrecognized archive chunk")
            data = path.read_bytes()
            metadata = json.loads(metadata_path.read_text())
            if (len(data) != end - start + 1 or original.sha256(data) != metadata["sha256"]
                    or metadata["start"] != start or metadata["end"] != end):
                raise ValueError("archive chunk checksum or range mismatch")
        else:
            data = self.client.range(self.identity, start, end)
            original.immutable_write(metadata_path, original.json_bytes({
                "start": start, "end": end, "size": len(data), "sha256": original.sha256(data),
            }))
            original.immutable_write(path, data)
            self.downloaded_bytes += len(data)
        self.cached_index, self.cached_data = index, data
        return data

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size > 8_000_000:
            raise ValueError("unexpectedly large ZIP read")
        result = []
        while size > 0:
            index, offset = divmod(self.position, CHUNK_SIZE)
            data = self.chunk(index)
            length = min(size, len(data) - offset)
            result.append(data[offset:offset + length])
            self.position += length
            size -= length
        return b"".join(result)

    def archive_sha256(self):
        digest = hashlib.sha256()
        for index in range((self.size + CHUNK_SIZE - 1) // CHUNK_SIZE):
            digest.update(self.chunk(index))
        return digest.hexdigest()


def probe_video(path: Path, timestamps: list[float]) -> dict:
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time:"
        "stream=codec_name,width,height,nb_frames,duration,time_base,r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ]))
    presentation = [float(frame["best_effort_timestamp_time"]) * 1000 for frame in probe["frames"]]
    if len(presentation) != len(timestamps):
        raise ValueError(f"encoded frame count changed: {len(presentation)} != {len(timestamps)}")
    errors = [actual - (expected - timestamps[0])
              for actual, expected in zip(presentation, timestamps, strict=True)]
    if max(map(abs, errors), default=0) > 1.001:
        raise ValueError("encoded timestamps differ from official sync by more than 1 ms")
    if any(right <= left for left, right in zip(presentation, presentation[1:], strict=False)):
        raise ValueError("encoded timestamps do not increase")
    return {"stream": probe["streams"][0], "frame_count": len(presentation),
            "presentation_timestamps_ms": presentation, "max_abs_timestamp_error_ms": max(map(abs, errors)),
            "mean_abs_timestamp_error_ms": sum(map(abs, errors)) / len(errors)}


def make_video(directory: Path, filenames: list[str], timestamps: list[float]) -> dict:
    if len(filenames) < 2:
        raise ValueError("continuous replay requires at least two synchronized frames")
    lines = ["ffconcat version 1.0"]
    for index, name in enumerate(filenames):
        if PurePosixPath(name).name != name or "'" in name:
            raise ValueError("unsafe concat filename")
        lines.extend([f"file '{name}'", "option framerate 1000"])
        if index + 1 < len(timestamps):
            lines.append(f"duration {(timestamps[index + 1] - timestamps[index]) / 1000:.9f}")
    concat = directory / "frames.ffconcat"
    original.immutable_write(concat, ("\n".join(lines) + "\n").encode())
    video = directory / "original-rgb.mp4"
    metadata_path = directory / "video.json"
    if video.exists():
        if video.is_symlink() or not metadata_path.is_file():
            raise ValueError("unrecognized existing video")
        previous = json.loads(metadata_path.read_text())
        if original.sha256(video.read_bytes()) != previous["sha256"]:
            raise ValueError("existing video checksum mismatch")
        probe_video(video, timestamps)
        return previous
    descriptor, temporary = tempfile.mkstemp(prefix=".urfd-video-", suffix=".mp4", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-enc_time_base", "1:1000",
        "-video_track_timescale", "1000000", "-movflags", "+faststart", "-threads", "1",
        "-filter_threads", "1", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        verification = probe_video(temporary, timestamps)
        data = temporary.read_bytes()
        metadata = {
            "path": str(video.name), "sha256": original.sha256(data), "bytes": len(data),
            "encoding": "H.264 VFR from every synchronized original RGB frame; no overlays",
            "timestamp_basis": "official milliseconds normalized to the first retained source frame",
            "time_base_ms": 1,
            "end_duration_note": "The final frame has no following official timestamp; its encoded display "
                                 "duration is determined by the 1 ms encoder time base, not an inferred sample.",
            "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            **verification,
        }
        original.immutable_write(metadata_path, original.json_bytes(metadata))
        original.immutable_write(video, data)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def prepare_sequence(root: Path, sequence: str) -> dict:
    planned = next(sample for sample in original.split_plan() if sample["sequence_id"] == sequence)
    if planned["split"] != "val" or sequence not in SEQUENCES:
        raise ValueError("only the two preselected validation sequences may be prepared")
    directory = root / DATA_PATH / sequence
    if directory.is_symlink():
        raise ValueError("refusing symlink output directory")
    directory.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    with original.HttpClient() as client:
        identity = client.head(f"{original.BASE_URL}data/{sequence}-cam0-rgb.zip")
        marker = directory / "archive.json"
        if not marker.exists() and any(directory.iterdir()):
            raise ValueError("unrecognized nonempty output directory")
        original.immutable_write(marker, original.json_bytes(identity))
        print(f"source {sequence} archive_bytes={identity['content_length']}", flush=True)
        sync_url = f"{original.BASE_URL}data/{sequence}-data.csv"
        sync_data = original.cached_small(client, sync_url, directory / "sync.csv")
        times = original.synchronization(sync_data)
        remote = ChunkedArchive(client, identity, directory / "archive-parts")
        with zipfile.ZipFile(remote) as archive:
            members = original.png_members(archive, sequence)
            retained = [(number, info) for number, info in members if number in times]
            excluded = [number for number, _ in members if number not in times]
            if not retained:
                raise ValueError("no synchronized RGB frames")
            allowed = {PurePosixPath(info.filename).name for _, info in retained} | {
                "archive.json", "archive-parts", "sync.csv", "sync.csv.json", "metadata.json",
                "frames.ffconcat", "original-rgb.mp4", "video.json",
            }
            unknown = [path.name for path in directory.iterdir() if path.name not in allowed
                       and not path.name.startswith((".urfd-write-", ".urfd-video-"))]
            if unknown:
                raise ValueError(f"unrecognized files: {unknown}")
            provenance, dimensions = [], set()
            for index, (number, info) in enumerate(retained):
                path = directory / PurePosixPath(info.filename).name
                if path.is_symlink():
                    raise ValueError("refusing symlink PNG")
                data = path.read_bytes() if path.exists() else archive.read(info)
                dimensions.add(original.validate_png(data, info))
                original.immutable_write(path, data)
                provenance.append({"frame_number": number, "timestamp_ms": times[number],
                                   "path": path.relative_to(root).as_posix(), "member_name": info.filename,
                                   "crc32": f"{info.CRC:08x}", "sha256": original.sha256(data),
                                   "size": len(data)})
                if (index + 1) % 50 == 0 or index + 1 == len(retained):
                    print(f"frames {sequence} {index + 1}/{len(retained)} "
                          f"elapsed={time.monotonic() - tick:.1f}s", flush=True)
            if len(dimensions) != 1:
                raise ValueError("dimensions changed within sequence")
            archive_hash = remote.archive_sha256()
        width, height = dimensions.pop()
        timestamps = [item["timestamp_ms"] for item in provenance]
        video = make_video(directory, [Path(item["path"]).name for item in provenance], timestamps)
        result = {
            "schema_version": "1.0", "dataset": "urfd-continuous-v1", **planned,
            "selection_basis": "two validation sequence IDs fixed before this continuous retrieval",
            "source_url": identity["url"], "archive_size": identity["content_length"],
            "archive_etag": identity["etag"], "archive_last_modified": identity["last_modified"],
            "archive_sha256": archive_hash, "member_count": len(members), "frame_count": len(provenance),
            "excluded_unsynchronized_frame_numbers": excluded, "width": width, "height": height,
            "frames": [item["path"] for item in provenance],
            "frame_numbers": [item["frame_number"] for item in provenance],
            "frame_timestamps_ms": timestamps, "members": provenance,
            "sync_url": sync_url, "sync_sha256": original.sha256(sync_data),
            "license": "CC-BY-NC-SA-4.0", "source_page": original.BASE_URL + "uf.html",
            "timestamp_policy": "Retain every RGB frame with an official synchronization row; no FPS guessing",
            "inference_input_note": "Use original PNGs and frame_timestamps_ms for exact streaming inference; "
                                    "the MP4 is a lossy, timestamp-verified VFR replay.",
            "video": {**video, "path": video_path(root, directory / video["path"])},
        }
        original.immutable_write(directory / "metadata.json", original.json_bytes(result))
        print(f"complete {sequence} frames={len(provenance)} elapsed={time.monotonic() - tick:.1f}s "
              f"downloaded_bytes_this_run={remote.downloaded_bytes}", flush=True)
        return result


def video_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required for verified original-frame MP4 output")
    import cv2

    cv2.setNumThreads(1)
    root = args.project_root.resolve()
    directory = root / DATA_PATH
    if directory.is_symlink():
        raise ValueError("refusing symlink dataset directory")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "dataset.json"
    if not marker.exists() and any(directory.iterdir()):
        raise ValueError("refusing unrecognized dataset directory")
    original.immutable_write(marker, original.json_bytes({
        "schema_version": "1.0", "dataset": "urfd-continuous-v1", "sequences": list(SEQUENCES),
        "license": "CC-BY-NC-SA-4.0", "purpose": "full synchronized validation streams, not 16-frame repeats",
    }))

    def prepare(sequence):
        for attempt in range(3):
            try:
                return prepare_sequence(root, sequence)
            except Exception as error:
                print(f"retry {sequence} {attempt + 1}/3: {error}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare, SEQUENCES))
    original.immutable_write(directory / "manifest.json", original.json_bytes({
        "schema_version": "1.0", "dataset": "urfd-continuous-v1", "samples": results,
    }))


if __name__ == "__main__":
    main()
