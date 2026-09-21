"""Fetch 16 deterministic cam0 RGB frames per URFD sequence using HTTP ranges.

Raw archives remain on the author's server. Existing final files are immutable:
resumption validates their identity, CRC and decoded PNG contents before reuse.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import tempfile
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/"
SEED = 20260921
FRAME_COUNT = 16
DATA_PATH = Path("datasets/private/urfd-cam0")
MANIFEST_PATH = Path("datasets/manifests/urfd-sequences-v1.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def immutable_write(path: Path, data: bytes) -> None:
    """Publish atomically without ever replacing an existing final file."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"existing file differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".urfd-write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"concurrent file differs: {path}") from None
    finally:
        os.unlink(temporary)


def split_plan() -> list[dict]:
    samples = []
    for prefix, count, label, train, val in (("fall", 30, 1, 18, 6), ("adl", 40, 0, 24, 8)):
        ids = [f"{prefix}-{index:02d}" for index in range(1, count + 1)]
        random.Random(SEED).shuffle(ids)
        for index, sequence in enumerate(ids):
            samples.append({"sequence_id": sequence, "group_id": f"urfd:{sequence}", "label": label,
                            "split": "train" if index < train else "val" if index < train + val else "test"})
    return sorted(samples, key=lambda sample: sample["sequence_id"])


def sample_indices(count: int, requested: int = FRAME_COUNT) -> list[int]:
    if count < requested or requested < 1:
        raise ValueError("sequence must contain at least the requested number of frames")
    return [(2 * index + 1) * count // (2 * requested) for index in range(requested)]


class HttpClient:
    def __init__(self):
        # requests is supplied by the local vision environment. Keep a stdlib
        # fallback for standalone use; both paths enforce the same Range checks.
        try:
            import requests
        except ImportError:
            self._session = None
        else:
            self._session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._session is not None:
            self._session.close()

    def _open(self, url: str, *, method: str = "GET", headers: dict | None = None):
        if not url.startswith("https://"):
            raise ValueError("only HTTPS sources are allowed")
        request_headers = {
            "User-Agent": "campus-safety-urfd-research/1.0", "Accept-Encoding": "identity", **(headers or {}),
        }
        if self._session is None:
            request = urllib.request.Request(url, method=method, headers=request_headers)
            response = urllib.request.urlopen(request, timeout=40)
        else:
            response = _SessionResponse(self._session.request(
                method, url, headers=request_headers, timeout=(10, 40), stream=True,
            ))
        if not response.geturl().startswith("https://"):
            response.close()
            raise ValueError("refusing non-HTTPS redirect")
        return response

    def head(self, url: str) -> dict:
        with self._open(url, method="HEAD") as response:
            if response.status != 200:
                raise ValueError(f"HEAD returned {response.status}")
            size = int(response.headers["Content-Length"])
            if size <= 0:
                raise ValueError("invalid archive size")
            result = {"url": url, "content_length": size, "etag": response.headers.get("ETag"),
                      "last_modified": response.headers.get("Last-Modified")}
            if not result["etag"] and not result["last_modified"]:
                raise ValueError("archive has no HTTP identity validator")
            return result

    def range(self, identity: dict, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}",
                   "If-Range": identity.get("etag") or identity["last_modified"]}
        with self._open(identity["url"], headers=headers) as response:
            # Inspect status before reading: never download a full ignored range.
            if response.status != 206:
                raise ValueError(f"range request returned {response.status}; refusing full response")
            expected = f"bytes {start}-{end}/{identity['content_length']}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("incorrect Content-Range")
            for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
                if identity.get(key) and response.headers.get(header) != identity[key]:
                    raise ValueError("archive HTTP identity changed during download")
            expected_size = end - start + 1
            if int(response.headers.get("Content-Length", expected_size)) != expected_size:
                raise ValueError("incorrect range Content-Length")
            data = response.read(expected_size + 1)
            if len(data) != expected_size:
                raise ValueError("truncated or oversized range response")
            return data

    def small(self, url: str, limit: int = 2_000_000) -> tuple[bytes, dict]:
        with self._open(url) as response:
            if response.status != 200:
                raise ValueError(f"small file returned {response.status}")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise ValueError("small file exceeds download limit")
            data = response.read(limit + 1)
            if len(data) > limit or (declared is not None and len(data) != int(declared)):
                raise ValueError("small file length mismatch")
            return data, {"url": url, "content_length": len(data), "sha256": sha256(data),
                          "etag": response.headers.get("ETag"),
                          "last_modified": response.headers.get("Last-Modified")}


class _SessionResponse:
    def __init__(self, response):
        self.response = response
        self.status, self.headers = response.status_code, response.headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def geturl(self):
        return self.response.url

    def read(self, size):
        return self.response.raw.read(size, decode_content=False)

    def close(self):
        self.response.close()


class RangeZipFile(io.RawIOBase):
    def __init__(self, client: HttpClient, identity: dict, tail: bytes):
        self.client, self.identity = client, identity
        self.size = identity["content_length"]
        self.position = 0
        self.tail = (self.size - len(tail), tail)
        self.segment: tuple[int, bytes] | None = None
        self.network_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("negative ZIP seek")
        return self.position

    def prefetch(self, start: int, length: int) -> None:
        if length <= 0 or length > 8_000_000:
            raise ValueError("unexpectedly large ZIP member")
        end = min(self.size, start + length) - 1
        data = self.client.range(self.identity, start, end)
        self.network_bytes += len(data)
        self.segment = (start, data)

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        for cached in (self.tail, self.segment):
            if cached is not None:
                start, data = cached
                offset = self.position - start
                if 0 <= offset and offset + size <= len(data):
                    self.position += size
                    return data[offset:offset + size]
        self.prefetch(self.position, size)
        return self.read(size)


def png_members(archive: zipfile.ZipFile, sequence: str) -> list[tuple[int, zipfile.ZipInfo]]:
    members = []
    seen = set()
    pattern = re.compile(rf"{re.escape(sequence)}-cam0-rgb/{re.escape(sequence)}-cam0-rgb-(\d+)\.png")
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1):
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        if info.is_dir():
            if info.filename != f"{sequence}-cam0-rgb/":
                raise ValueError("unexpected ZIP directory")
            continue
        match = pattern.fullmatch(info.filename)
        if match is None or info.file_size > 8_000_000:
            raise ValueError(f"unexpected ZIP member: {info.filename}")
        number = int(match.group(1))
        if number < 1 or number in seen:
            raise ValueError("duplicate or invalid PNG frame number")
        seen.add(number)
        members.append((number, info))
    return sorted(members, key=lambda member: member[0])


def synchronization(data: bytes) -> dict[int, float]:
    result = {}
    for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
        if len(row) < 2:
            raise ValueError("incomplete synchronization row")
        number, timestamp = int(row[0]), float(row[1])
        if number < 1 or number in result or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("duplicate or invalid synchronization frame/time")
        result[number] = timestamp
    if not result:
        raise ValueError("empty synchronization data")
    times = [result[number] for number in sorted(result)]
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("synchronization times must increase")
    return result


def validate_png(data: bytes, info: zipfile.ZipInfo) -> tuple[int, int]:
    import cv2
    import numpy as np

    if len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
        raise ValueError("PNG length or ZIP CRC mismatch")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("selected member is not PNG")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("PNG is not a decodable color image")
    return int(image.shape[1]), int(image.shape[0])


def cached_small(client: HttpClient, url: str, path: Path) -> bytes:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata_path.exists():
        if path.is_symlink() or metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(f"unrecognized cached file: {path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata["url"] != url:
            raise ValueError("cached URL mismatch")
        if path.exists():
            data = path.read_bytes()
            if len(data) != metadata["content_length"] or sha256(data) != metadata["sha256"]:
                raise ValueError("cached small-file checksum mismatch")
            return data
        data, fresh = client.small(url)
        if fresh["sha256"] != metadata["sha256"]:
            raise ValueError("small file changed while resuming")
    else:
        data, metadata = client.small(url)
        immutable_write(metadata_path, json_bytes(metadata))
    immutable_write(path, data)
    return data


def prepare_sequence(project_root: Path, planned: dict, client: HttpClient) -> dict:
    sequence = planned["sequence_id"]
    directory = project_root / DATA_PATH / sequence
    if directory.is_symlink():
        raise ValueError("refusing symlink sequence directory")
    directory.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}data/{sequence}-cam0-rgb.zip"
    identity = client.head(url)
    identity_path = directory / "archive.json"
    # Archive identity is written before any resumable data is accepted.
    if not identity_path.exists() and any(directory.iterdir()):
        raise ValueError(f"unrecognized nonempty sequence directory: {directory}")
    immutable_write(identity_path, json_bytes(identity))
    tail_path = directory / "zip-directory.bin"
    tail_length = min(identity["content_length"], 65_557)
    tail_meta_path = directory / "zip-directory.json"
    if tail_path.exists():
        if tail_path.is_symlink() or not tail_meta_path.is_file():
            raise ValueError("unrecognized ZIP directory cache")
        tail = tail_path.read_bytes()
        tail_meta = json.loads(tail_meta_path.read_text())
        if len(tail) != tail_length or sha256(tail) != tail_meta["sha256"]:
            raise ValueError("cached ZIP directory checksum mismatch")
    else:
        tail = client.range(identity, identity["content_length"] - tail_length, identity["content_length"] - 1)
        immutable_write(tail_meta_path, json_bytes({"sha256": sha256(tail), "length": len(tail)}))
        immutable_write(tail_path, tail)
    remote = RangeZipFile(client, identity, tail)
    with zipfile.ZipFile(remote) as archive:
        members = png_members(archive, sequence)
        immutable_write(directory / "zip-index.json", json_bytes([
            {"frame_number": number, "name": info.filename, "crc32": f"{info.CRC:08x}",
             "size": info.file_size, "compressed_size": info.compress_size, "header_offset": info.header_offset}
            for number, info in members
        ]))
        sync_url = f"{BASE_URL}data/{sequence}-data.csv"
        sync_data = cached_small(client, sync_url, directory / "sync.csv")
        times = synchronization(sync_data)
        missing = [number for number, _ in members if number not in times]
        population = [(number, info) for number, info in members if number in times]
        if len(population) < FRAME_COUNT:
            raise ValueError("missing synchronization timestamps leave fewer than 16 RGB frames")
        chosen = [population[index] for index in sample_indices(len(population))]
        names = {PurePosixPath(info.filename).name for _, info in chosen}
        allowed = names | {"archive.json", "zip-directory.bin", "zip-directory.json", "zip-index.json", "sync.csv",
                           "sync.csv.json", "sample.json"}
        unknown = [path.name for path in directory.iterdir()
                   if path.name not in allowed and not path.name.startswith(".urfd-write-")]
        if unknown:
            raise ValueError(f"unrecognized files; refusing overwrite: {unknown}")
        paths, provenance, dimensions = [], [], set()
        for number, info in chosen:
            path = directory / PurePosixPath(info.filename).name
            if path.is_symlink():
                raise ValueError("refusing symlink PNG")
            if path.exists():
                data = path.read_bytes()
            else:
                # Include local-header variable fields; selected member data stays
                # in one bounded RAM segment and is replaced on the next frame.
                remote.prefetch(info.header_offset, 30 + len(info.filename.encode()) + len(info.extra)
                                + info.compress_size + 1024)
                data = archive.read(info)
            dimensions.add(validate_png(data, info))
            immutable_write(path, data)
            paths.append(path.relative_to(project_root).as_posix())
            provenance.append({"name": info.filename, "frame_number": number,
                               "crc32": f"{info.CRC:08x}", "sha256": sha256(data),
                               "size": info.file_size, "compressed_size": info.compress_size,
                               "header_offset": info.header_offset})
        if len(dimensions) != 1:
            raise ValueError("image dimensions changed within sequence")
        width, height = dimensions.pop()
        sample = {**planned, "frames": paths, "frame_numbers": [number for number, _ in chosen],
                  "frame_timestamps_ms": [times[number] for number, _ in chosen],
                  "source_url": url, "archive_size": identity["content_length"],
                  "archive_etag": identity["etag"], "archive_last_modified": identity["last_modified"],
                  "member_count": len(members), "width": width, "height": height,
                  "sync_url": sync_url, "sync_sha256": sha256(sync_data),
                  "sampling": "16 equal-bin centers over sorted original RGB frame numbers",
                  "members": provenance, "license": "CC-BY-NC-SA-4.0",
                  "label_basis": "original fall/ADL sequence category; not depth posture labels"}
        if missing:
            sample.update({
                "sampling": "16 equal-bin centers over sorted RGB frame numbers with official sync timestamps",
                "sampling_population_count": len(population),
                "excluded_unsynchronized_frame_numbers": missing,
                "data_quality_note": "Original RGB frames without official synchronization rows were excluded "
                                     "before deterministic sampling; no timestamps were interpolated or inferred.",
            })
        immutable_write(directory / "sample.json", json_bytes(sample))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sequence", action="append", help="prepare a pilot; no partial manifest is published")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.retries <= 5:
        parser.error("workers must be 1..4 and retries 1..5")
    import cv2

    cv2.setNumThreads(1)
    root = args.project_root.resolve()
    directory = root / DATA_PATH
    if directory.is_symlink():
        raise ValueError("refusing symlink dataset directory")
    directory.mkdir(parents=True, exist_ok=True)
    plan = split_plan()
    descriptor = {"schema_version": "1.0", "dataset": "urfd-cam0", "seed": SEED,
                  "frame_count": FRAME_COUNT, "class_names": ["non_fall", "fall"],
                  "license": "CC-BY-NC-SA-4.0", "source_page": BASE_URL + "uf.html"}
    ownership = directory / "dataset.json"
    if not ownership.exists() and any(directory.iterdir()):
        raise ValueError("refusing unrecognized nonempty dataset directory")
    immutable_write(ownership, json_bytes(descriptor))
    immutable_write(directory / "split-plan.json", json_bytes(plan))
    with HttpClient() as client:
        page = cached_small(client, BASE_URL + "uf.html", directory / "license-page.html")
    if b"Attribution-NonCommercial-ShareAlike 4.0" not in page:
        raise ValueError("official license statement not found in snapshot")
    if args.sequence:
        requested = set(args.sequence)
        if not requested <= {sample["sequence_id"] for sample in plan}:
            raise ValueError("unknown requested sequence")
        selected = [sample for sample in plan if sample["sequence_id"] in requested]
    else:
        selected = plan

    def prepare(planned):
        for attempt in range(args.retries):
            try:
                with HttpClient() as client:
                    return prepare_sequence(root, planned, client)
            except Exception as error:
                print(f"retry {planned['sequence_id']} {attempt + 1}/{args.retries}: {error}", flush=True)
                if attempt + 1 == args.retries:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    completed, failures = [], []
    tick = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, sample): sample["sequence_id"] for sample in selected}
        for future in concurrent.futures.as_completed(futures):
            try:
                sample = future.result()
                completed.append(sample)
                print(f"ready {len(completed)}/{len(selected)} {sample['sequence_id']} "
                      f"{sample['split']} frames={len(sample['frames'])} "
                      f"elapsed={time.monotonic() - tick:.1f}s", flush=True)
            except Exception as error:
                failures.append((futures[future], str(error)))
    if failures:
        raise RuntimeError(f"incomplete dataset; rerun to resume: {failures}")
    if len(completed) == 70:
        manifest = {**descriptor, "license_snapshot": (DATA_PATH / "license-page.html").as_posix(),
                    "license_snapshot_sha256": sha256(page),
                    "sampling_policy": "16 equal-bin centers over each sequence's RGB frame numbers "
                                       "intersected with official synchronization frame numbers",
                    "samples": sorted(completed, key=lambda sample: sample["sequence_id"])}
        immutable_write(root / MANIFEST_PATH, json_bytes(manifest))
        print(f"complete manifest -> {root / MANIFEST_PATH}", flush=True)
    else:
        print("pilot complete; final manifest withheld until all 70 sequences are verified", flush=True)


if __name__ == "__main__":
    main()
