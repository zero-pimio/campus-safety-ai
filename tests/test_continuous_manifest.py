from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from campus_safety_ai.training.continuous_manifest import FIELDS, audit_continuous_manifest


class ContinuousManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "sessions.csv"

    def rows(self) -> list[dict[str, str]]:
        result = []
        for split in ("train", "val", "test"):
            video = self.root / f"{split}.mp4"
            video.write_bytes(f"synthetic fixture {split}".encode())
            annotation = self.root / f"{split}.json"
            annotation.write_text('{"sessions": [], "events": []}')
            result.append(dict(zip(FIELDS, (
                split, video.name, split, f"person-{split}", f"camera-{split}", f"recording-{split}",
                "normal", "60", annotation.name, hashlib.sha256(video.read_bytes()).hexdigest(),
            ), strict=True)))
        return result

    def audit(self, rows: list[dict[str, str]], *, strict: bool = False) -> dict:
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return audit_continuous_manifest(self.manifest, project_root=self.root, strict=strict)

    def test_complete_manifest_verifies_bytes_but_reports_behavior_gaps(self) -> None:
        report = self.audit(self.rows(), strict=True)
        self.assertTrue(report["passed"])
        self.assertTrue(report["eligible_for_person_camera_disjoint_evaluation"])
        self.assertEqual(report["file_hashes_verified"], 3)
        self.assertEqual(report["totals"]["independent_source_groups"], 3)
        self.assertIn("squatting", report["by_split"]["test"]["missing_behaviors"])
        self.assertEqual(report["totals"]["listed_clip_duration_seconds"], 180)

    def test_person_camera_group_hash_and_path_leakage_are_blocked(self) -> None:
        for field, code in (
            ("person_ids", "person_cross_split"), ("camera_id", "camera_cross_split"),
            ("source_group", "source_group_cross_split"), ("sha256", "sha256_cross_split"),
            ("path", "path_cross_split"),
        ):
            with self.subTest(field=field):
                rows = self.rows()
                rows[1][field] = rows[0][field]
                report = self.audit(rows)
                self.assertFalse(report["passed"])
                self.assertIn(code, {error["code"] for error in report["errors"]})

    def test_all_people_are_checked_and_ids_ignore_case_and_whitespace(self) -> None:
        rows = self.rows()
        rows[0]["person_ids"] = " first ; Shared-Actor "
        rows[2]["person_ids"] = "second;shared-actor"
        report = self.audit(rows)
        self.assertIn("person_cross_split", {error["code"] for error in report["errors"]})

    def test_unknown_id_prevents_independence_claim_and_strict_admission(self) -> None:
        for field, value, code in (
            ("person_ids", "known;UNKNOWN", "unknown_person"),
            ("person_ids", "", "unknown_person"), ("camera_id", "N/A", "unknown_camera"),
        ):
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                draft = self.audit(rows)
                self.assertTrue(draft["passed"])
                self.assertFalse(draft["identities_complete"])
                self.assertFalse(draft["eligible_for_person_camera_disjoint_evaluation"])
                report = self.audit(rows, strict=True)
                self.assertIn(code, {error["code"] for error in report["errors"]})

    def test_strict_hashes_cannot_hide_duplicate_video_with_falsified_digest(self) -> None:
        rows = self.rows()
        (self.root / rows[1]["path"]).write_bytes((self.root / rows[0]["path"]).read_bytes())
        report = self.audit(rows, strict=True)
        codes = {error["code"] for error in report["errors"]}
        self.assertIn("sha256_mismatch", codes)
        self.assertIn("verified_sha256_cross_split", codes)

    def test_multiple_clips_in_one_recording_count_once(self) -> None:
        rows = self.rows()
        derived = dict(rows[0], clip_id="train-later", path="train-later.mp4", sha256="a" * 64,
                       behavior="bending")
        report = self.audit([*rows, derived])
        self.assertTrue(report["passed"])
        self.assertEqual(report["totals"]["clips"], 4)
        self.assertEqual(report["totals"]["independent_source_groups"], 3)
        self.assertEqual(report["by_split"]["train"]["independent_source_groups"], 1)

    def test_duplicate_content_cannot_inflate_source_group_count(self) -> None:
        rows = self.rows()
        rows.append(dict(rows[0], clip_id="duplicated", source_group="pretend-new-recording"))
        report = self.audit(rows)
        self.assertIn("duplicate_content_groups", {error["code"] for error in report["errors"]})

    def test_strict_missing_files_or_hash_are_rejected(self) -> None:
        rows = self.rows()
        (self.root / rows[0]["path"]).unlink()
        (self.root / rows[1]["annotation_path"]).unlink()
        rows[2]["sha256"] = ""
        report = self.audit(rows, strict=True)
        self.assertFalse(report["passed"])
        self.assertTrue({"missing_video", "missing_annotation_file", "missing_sha256"}.issubset(
            {error["code"] for error in report["errors"]}
        ))

    def test_invalid_labels_split_duration_and_source_group_fail_closed(self) -> None:
        for field, value, code in (
            ("split", "validation", "invalid_split"), ("behavior", "non_fall", "invalid_behavior"),
            ("duration_seconds", "nan", "invalid_duration"),
            ("duration_seconds", "-1", "invalid_duration"),
            ("source_group", "unknown", "missing_field"),
            ("sha256", "xxx", "invalid_sha256"),
        ):
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                report = self.audit(rows)
                self.assertIn(code, {error["code"] for error in report["errors"]})
                self.assertFalse(report["passed"])

    def test_empty_template_never_passes_as_a_collected_dataset(self) -> None:
        report = self.audit([])
        self.assertFalse(report["passed"])
        self.assertEqual(report["totals"]["clips"], 0)
        self.assertFalse(report["eligible_for_person_camera_disjoint_evaluation"])

    def test_symlink_alias_cannot_hide_same_path_across_splits(self) -> None:
        rows = self.rows()
        alias = self.root / "alias.mp4"
        alias.symlink_to(self.root / rows[0]["path"])
        rows[1]["path"] = alias.name
        report = self.audit(rows)
        self.assertIn("path_cross_split", {error["code"] for error in report["errors"]})

    def test_cli_writes_failure_report_and_never_replaces_existing_report(self) -> None:
        self.audit([])
        output = self.root / "audit.json"
        command = [sys.executable, "-m", "campus_safety_ai.apps.audit_continuous_manifest",
                   "--manifest", str(self.manifest), "--project-root", str(self.root),
                   "--output", str(output)]
        first = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(first.returncode, 1)
        saved = output.read_bytes()
        self.assertFalse(json.loads(saved)["passed"])
        second = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn("output already exists", second.stderr)
        self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
