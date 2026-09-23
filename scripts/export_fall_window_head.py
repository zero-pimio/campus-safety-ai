#!/usr/bin/env python3
"""Export a frozen fall classification head and verify validation-only CPU parity."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from campus_safety_ai.adapters.runtimes.pose_fall import FrozenPoseFallHead, sha256
from campus_safety_ai.training.pose_motion import FEATURE_NAMES


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export(checkpoint: Path, *, development_windows: Path | None = None,
           output_dir: Path | None = None) -> dict:
    """No test data, detector export, quantization, or deployment is performed."""
    checkpoint = checkpoint.resolve()
    development_windows = (development_windows or checkpoint.parent / "development-windows.json").resolve()
    output_dir = (output_dir or checkpoint.parent / "onnx").resolve()
    if output_dir.exists():
        raise ValueError("export output directory already exists; choose a new directory")
    paths = {"checkpoint": checkpoint, "protocol": checkpoint.parent / "protocol.json",
             "frozen": checkpoint.parent / "frozen.json", "development_windows": development_windows}
    original_hashes = {name: sha256(path) for name, path in paths.items()}
    head = FrozenPoseFallHead(checkpoint, "cpu")
    protocol = json.loads(paths["protocol"].read_text())
    frozen = json.loads(paths["frozen"].read_text())
    windows = json.loads(development_windows.read_text())
    if (windows["window_policy"] != protocol["window_policy"]
            or protocol["window_policy"] != head.extraction_identity.get("window_policy")):
        raise ValueError("development windows differ from the frozen window policy")
    # Reject a mixed file outright; never run the head on test/external records.
    if any(row["split"] not in {"train", "val"} for row in windows["rows"]):
        raise ValueError("export parity requires a development-only train/val window file")
    selected = [row for row in windows["rows"] if row["split"] == "val" and row["features"] is not None]
    if not selected:
        raise ValueError("export parity requires observed validation features")
    if any(row["sample_id"] not in protocol["cache_sha256"] for row in selected):
        raise ValueError("validation sample is absent from the frozen development cache index")
    features = np.asarray([row["features"] for row in selected], dtype=np.float32)
    if features.shape != (len(selected), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("validation features must be finite float32 motion descriptors of the frozen dimension")
    all_features = np.concatenate([features, np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)])

    import onnx
    import onnxruntime as ort

    output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path, report_path = output_dir / "model.onnx", output_dir / "export.json"
    report = {
        "schema_version": "1.0", "status": "running", "created_at": datetime.now(UTC).isoformat(),
        "input_sha256": original_hashes, "checkpoint_sha256": head.checkpoint_sha256,
        "model_version": head.model_version, "pose_detector_sha256": head.pose_checkpoint_sha256,
        "window_policy": protocol["window_policy"], "threshold": head.threshold,
        "event_policy": frozen["event_policy"], "opset": 17, "dynamic_batch": True,
        "input": {"name": "motion_features", "shape": ["batch", len(FEATURE_NAMES)], "dtype": "float32",
                  "meaning": "raw unstandardized motion descriptor; trained normalization is inside this head"},
        "output": {"name": "logits", "shape": ["batch", 2], "dtype": "float32",
                   "class_names": ["no_fall_transition", "fall_transition"],
                   "score": "softmax(logits)[:, 1]; apply the frozen event policy separately"},
        "validation_feature_count": len(features), "raw_zero_control_count": 1,
        "validation_sample_ids": sorted({row["sample_id"] for row in selected}),
        "exporter_sha256": sha256(Path(__file__)),
        "scope": "Experimental classification head only, including trained feature normalization. "
                 "Excludes YOLO, video decoding, pose association, motion descriptors and event analysis. "
                 "CPU ONNX Runtime numerical parity is not RKNN/RK3588 support, speed, precision, "
                 "long-run stability, or production promotion.",
    }
    write_report(report_path, report)
    try:
        torch.onnx.export(head._model, torch.from_numpy(all_features[:1]), onnx_path,
                          input_names=["motion_features"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"motion_features": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
        graph = onnx.load(onnx_path)
        onnx.checker.check_model(graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        report["onnx_sha256"] = sha256(onnx_path)
        report["onnxruntime_version"] = ort.__version__
        report["torch_version"] = str(torch.__version__)
        report["providers"] = session.get_providers()
        # Verify every feature once as a single sample and once in a batch.
        # This catches a falsely static batch axis as well as numerical drift.
        comparisons = []
        for batch_size in sorted({1, min(64, len(all_features))}):
            max_logit_error = max_probability_error = 0.0
            argmax_mismatches = threshold_mismatches = 0
            zero_result = None
            for start in range(0, len(all_features), batch_size):
                inputs = all_features[start:start + batch_size]
                with torch.inference_mode():
                    torch_logits = head._model(torch.from_numpy(inputs))
                    expected = torch_logits.numpy()
                    expected_scores = torch_logits.softmax(1)[:, 1].numpy()
                actual = session.run(["logits"], {"motion_features": inputs})[0]
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError("ONNX returned nonfinite or incorrectly shaped logits")
                probabilities = np.exp(actual.astype(np.float64) - actual.max(axis=1, keepdims=True))
                actual_scores = (probabilities / probabilities.sum(axis=1, keepdims=True))[:, 1]
                max_logit_error = max(max_logit_error, float(np.abs(expected - actual).max()))
                max_probability_error = max(max_probability_error, float(np.abs(expected_scores - actual_scores).max()))
                argmax_mismatches += int(np.count_nonzero(expected.argmax(1) != actual.argmax(1)))
                threshold_mismatches += int(np.count_nonzero((expected_scores >= head.threshold)
                                                            != (actual_scores >= head.threshold)))
                if start + len(inputs) == len(all_features):
                    zero_result = {"torch_score": float(expected_scores[-1]), "onnx_score": float(actual_scores[-1]),
                                   "below_frozen_end_threshold": bool(actual_scores[-1] < frozen["event_policy"]["end_score"])}
            comparisons.append({"batch_size": batch_size, "checked_features": len(all_features),
                                "maximum_logit_absolute_error": max_logit_error,
                                "maximum_probability_absolute_error": max_probability_error,
                                "argmax_mismatch_count": argmax_mismatches,
                                "frozen_threshold_mismatch_count": threshold_mismatches,
                                "raw_zero": zero_result})
        report["parity"] = {"comparison_runs": comparisons,
                            "maximum_logit_absolute_error": max(row["maximum_logit_absolute_error"] for row in comparisons),
                            "maximum_probability_absolute_error": max(row["maximum_probability_absolute_error"] for row in comparisons),
                            "argmax_agreement": all(row["argmax_mismatch_count"] == 0 for row in comparisons),
                            "frozen_threshold_agreement": all(row["frozen_threshold_mismatch_count"] == 0 for row in comparisons),
                            "absolute_logit_tolerance": 1e-4, "absolute_probability_tolerance": 1e-5}
        if any(row["maximum_logit_absolute_error"] > 1e-4
               or row["maximum_probability_absolute_error"] > 1e-5
               or row["argmax_mismatch_count"] or row["frozen_threshold_mismatch_count"] for row in comparisons):
            raise ValueError("ONNX numerical parity or prediction agreement failed")
        if original_hashes != {name: sha256(path) for name, path in paths.items()}:
            raise ValueError("frozen inputs changed during export")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_report(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-windows", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = export(args.checkpoint, development_windows=args.development_windows, output_dir=args.output_dir)
    print(json.dumps({"status": report["status"], "onnx_sha256": report["onnx_sha256"],
                      "validation_feature_count": report["validation_feature_count"], "parity": report["parity"]}))


if __name__ == "__main__":
    main()
