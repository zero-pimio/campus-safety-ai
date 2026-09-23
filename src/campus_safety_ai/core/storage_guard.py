"""Periodic, read-only disk pressure checks for one explicit runtime directory.

These checks reserve no disk space and do not promise unlimited runtime. Stop
producing new observations when blocked; finalization may still need the reserve.
"""
from __future__ import annotations

import copy
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoragePolicy:
    min_free_bytes: int = 256 * 1024 * 1024
    max_run_bytes: int = 2 * 1024 * 1024 * 1024
    check_every_frames: int = 30

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class StorageGuardError(RuntimeError):
    def __init__(self, message: str, report: dict):
        self.report = copy.deepcopy(report)
        super().__init__(message)


class StorageCapacityError(StorageGuardError):
    """Disk reserve or run size reached its limit; existing data was not changed."""


class StorageInspectionError(StorageGuardError):
    """Storage state cannot be established safely; fail closed instead of guessing."""


def _open_directory(path: Path) -> int:
    """Open all path components without traversing symlinks, including ancestors."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("run directory must be absolute and must not contain '..'")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _scan_directory(descriptor: int, depth: int = 0) -> dict:
    if depth > 64:
        raise ValueError("run directory nesting exceeds inspection limit")
    result = {"run_bytes": 0, "regular_files": 0, "directories": 1,
              "skipped_symlinks": 0, "skipped_special_files": 0, "vanished_entries": 0}
    with os.scandir(descriptor) as entries:
        for entry in entries:
            try:
                metadata = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                result["vanished_entries"] += 1
                continue
            if stat.S_ISLNK(metadata.st_mode):
                result["skipped_symlinks"] += 1
            elif stat.S_ISREG(metadata.st_mode):
                result["run_bytes"] += metadata.st_size
                result["regular_files"] += 1
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=descriptor)
                except FileNotFoundError:
                    result["vanished_entries"] += 1
                    continue
                try:
                    nested = _scan_directory(child, depth + 1)
                    for key, value in nested.items():
                        result[key] += value
                finally:
                    os.close(child)
            else:
                result["skipped_special_files"] += 1
    return result


class StorageGuard:
    """Check at startup and fixed frame intervals; reports never perform I/O.

    The supplied directory must exist and belong only to the current run. This
    class never discovers parent runtimes, deletes files, or follows symlinks.
    File sizes are logical bytes; hard-linked directory entries are each counted.
    """

    def __init__(self, run_directory: Path, policy: StoragePolicy | None = None) -> None:
        self.run_directory = Path(run_directory).absolute()
        self.policy = policy or StoragePolicy()
        if not isinstance(self.policy, StoragePolicy):
            raise TypeError("policy must be StoragePolicy")
        descriptor = _open_directory(self.run_directory)
        try:
            initial = os.fstat(descriptor)
            self._directory_identity = (initial.st_dev, initial.st_ino)
        finally:
            os.close(descriptor)
        self._last_checked_frame: int | None = None
        self._last_seen_frame: int | None = None
        self._report = {
            "status": "not_checked", "run_directory": str(self.run_directory),
            "policy": asdict(self.policy), "checks": 0, "checked_at": None,
            "processed_frames_checked": None, "next_check_at_frame": 0,
            "run_bytes": None, "disk_free_bytes": None, "violations": [],
            "size_semantics": "logical size of regular files only; symlinks not followed; "
                              "hard-linked entries counted individually; no deletion or disk reservation",
        }

    def report(self) -> dict:
        """Return the most recent snapshot without rescanning or mutable sharing."""
        return copy.deepcopy(self._report)

    def check(self, processed_frames: int = 0, *, force: bool = False) -> dict:
        if type(processed_frames) is not int or processed_frames < 0:
            raise ValueError("processed_frames must be a nonnegative integer")
        if self._last_seen_frame is not None and processed_frames < self._last_seen_frame:
            raise ValueError("processed_frames must not decrease within one run")
        self._last_seen_frame = processed_frames
        if (not force and self._report["status"] == "ok" and self._last_checked_frame is not None
                and processed_frames - self._last_checked_frame < self.policy.check_every_frames):
            return self.report()
        checked_at = time.time()
        report = {**self._report, "status": "checking", "checks": self._report["checks"] + 1,
                  "checked_at": checked_at, "processed_frames_checked": processed_frames,
                  "next_check_at_frame": processed_frames + self.policy.check_every_frames,
                  "violations": [], "run_bytes": None, "disk_free_bytes": None,
                  "regular_files": None, "directories": None, "skipped_symlinks": None,
                  "skipped_special_files": None, "vanished_entries": None, "error_type": None}
        descriptor = None
        try:
            descriptor = _open_directory(self.run_directory)
            initial = os.fstat(descriptor)
            if (initial.st_dev, initial.st_ino) != self._directory_identity:
                raise OSError("run directory was replaced since guard initialization")
            filesystem = os.fstatvfs(descriptor)
            report["disk_free_bytes"] = filesystem.f_bavail * filesystem.f_frsize
            report.update(_scan_directory(descriptor))
            # Validate that the named run has not been replaced during inspection.
            original = os.fstat(descriptor)
            current_descriptor = _open_directory(self.run_directory)
            try:
                current = os.fstat(current_descriptor)
            finally:
                os.close(current_descriptor)
            if (original.st_dev, original.st_ino) != (current.st_dev, current.st_ino):
                raise OSError("run directory changed during inspection")
        except (OSError, ValueError) as error:
            report["status"] = "failed"
            report["error_type"] = type(error).__name__
            report["violations"] = ["storage_inspection_failed"]
            self._report = report
            raise StorageInspectionError("cannot inspect current run storage safely; stop inference", report) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self._last_checked_frame = processed_frames
        if report["disk_free_bytes"] <= self.policy.min_free_bytes:
            report["violations"].append("disk_free_bytes_at_or_below_reserve")
        if report["run_bytes"] >= self.policy.max_run_bytes:
            report["violations"].append("run_bytes_at_or_above_limit")
        report["status"] = "blocked" if report["violations"] else "ok"
        self._report = report
        if report["violations"]:
            raise StorageCapacityError(
                "runtime storage limit reached; stop inference and preserve existing data: "
                f"free={report['disk_free_bytes']} bytes (reserve={self.policy.min_free_bytes}), "
                f"run={report['run_bytes']} bytes (limit={self.policy.max_run_bytes})", report)
        return self.report()
