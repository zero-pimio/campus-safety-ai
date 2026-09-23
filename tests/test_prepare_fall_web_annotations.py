"""Integrity checks for frozen public-data annotation preparation, without media/model inference."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_fall_web_annotations", ROOT / "scripts/prepare_fall_web_annotations.py")
assert SPEC and SPEC.loader
annotations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotations)


def test_decoded_pts_preserve_variable_frame_spacing_and_nonzero_origin():
    result = annotations.timing_from_probe({"frames": [
        {"best_effort_timestamp_time": "0.033333", "duration_time": "0.032"},
        {"best_effort_timestamp_time": "0.065333", "duration_time": "0.064"},
        {"best_effort_timestamp_time": "0.129333", "duration_time": "0.032"},
    ]})
    assert result["frame_timestamps_s"] == [0, 0.032, 0.096]
    assert result["frame_durations_s"] == [0.032, 0.064, 0.032]
    assert result["duration_s"] == 0.128
    assert result["first_pts_s"] == 0.033333


@pytest.mark.parametrize("timestamps", [[0, 0], [0, -1], [1, 2], [0, float("nan")], []])
def test_bad_timestamps_cannot_become_training_times(timestamps):
    with pytest.raises(ValueError):
        annotations.validate_times(timestamps)


def test_last_exposure_is_not_invented_from_average_fps():
    with pytest.raises(ValueError, match="explicit positive"):
        annotations.timing_from_probe({"frames": [{"best_effort_timestamp_time": "0"}]})


def test_up_fall_uses_actual_png_times_not_cfr_preview():
    data = {"original_frames": [
        {"name": "2018-07-04T12_00_00.000000.png", "timestamp": "2018-07-04T12:00:00.000000"},
        {"name": "2018-07-04T12_00_00.070000.png", "timestamp": "2018-07-04T12:00:00.070000"},
        {"name": "2018-07-04T12_00_00.100000.png", "timestamp": "2018-07-04T12:00:00.100000"},
    ]}
    result = annotations.original_png_timing(data)
    assert result["frame_timestamps_s"] == [0, 0.07, 0.10]
    assert result["duration_s"] == 0.10
    data["original_frames"][1]["timestamp"] = "2018-07-04T12:00:00.050000"
    with pytest.raises(ValueError, match="disagree"):
        annotations.original_png_timing(data)


def test_frozen_manifest_keeps_uncertainty_and_subject_groups():
    manifest = json.loads((ROOT / "datasets/manifests/fall-web-v1.json").read_text())
    samples = manifest["samples"]
    assert len(samples) == 25
    assert {split: sum(s["split"] == split for s in samples)
            for split in ("train", "val", "test", "external_diagnostic")} == {
                "train": 12, "val": 6, "test": 5, "external_diagnostic": 2}
    annotations.validate_splits(samples)
    gmd_falls = [s for s in samples if s["dataset"] == "GMDCSA24" and s["events"]]
    assert len(gmd_falls) == 7
    for sample in samples:
        annotations.validate_intervals(sample)
        if sample["dataset"] == "GMDCSA24":
            assert sample["camera_id"] is None
            assert sample["split"] == annotations.SPLITS[sample["subject_id"]]
        else:
            assert sample["split"] == "external_diagnostic"
            assert all(i["training_label"] is None for i in sample["intervals"])
    for sample in gmd_falls:
        motion = [i for i in sample["intervals"] if i["training_label"] == 1]
        assert len(motion) == 1
        assert motion[0]["end_s"] - motion[0]["start_s"] >= 0.2
        assert motion[0]["end_s"] < sample["timing"]["duration_s"]
        assert not sample["events"][0]["recovery_observed"]
        assert sample["events"][0]["recovery_time_s"] is None
        assert len(sample["evidence"]) == 2


def test_split_validator_catches_same_subject_and_source_crossing():
    samples = json.loads((ROOT / "datasets/manifests/fall-web-v1.json").read_text())["samples"]
    modified = copy.deepcopy(samples)
    modified[0]["split"] = "test"
    with pytest.raises(ValueError, match="crosses splits"):
        annotations.validate_splits(modified)
    modified = copy.deepcopy(samples)
    modified[0]["camera_id"] = "invented-camera-1"
    with pytest.raises(ValueError, match="unknown"):
        annotations.validate_splits(modified)


def test_uncertain_intervals_cannot_be_silently_labeled():
    samples = json.loads((ROOT / "datasets/manifests/fall-web-v1.json").read_text())["samples"]
    sample = copy.deepcopy(next(s for s in samples if s["events"]))
    next(i for i in sample["intervals"] if i["label"] == "uncertain")["training_label"] = 0
    with pytest.raises(ValueError, match="uncertain movement"):
        annotations.validate_intervals(sample)
