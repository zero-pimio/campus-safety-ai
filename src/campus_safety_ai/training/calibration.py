"""Decision thresholds fitted on validation scores, never on held-out test data."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from campus_safety_ai.training.metrics import classification_metrics


def validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number between 0 and 1")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1")
    return float(threshold)


def predictions_at_threshold(scores: list[float], threshold: float = 0.5) -> list[int]:
    threshold = validate_threshold(threshold)
    for score in scores:
        validate_threshold(score)
    return [int(score >= threshold) for score in scores]


def select_validation_threshold(
    labels: list[int], scores: list[float], *, split: str, partial: bool = False
) -> dict[str, Any]:
    """Maximize balanced accuracy; break ties by F1, then distance to 0.5.

    Candidate midpoints cover each attainable decision between distinct scores.
    Including 0.5 preserves the conventional threshold when equally effective.
    This is decision-threshold selection, not probability calibration.
    """
    if split != "val":
        raise ValueError("threshold selection requires the val split; never fit on test data")
    if partial:
        raise ValueError("threshold selection requires the complete validation split")
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        raise ValueError("threshold selection requires aligned scores and both binary labels")
    predictions_at_threshold(scores)
    unique = sorted(set(scores))
    midpoints = (
        left + (right - left) / 2 for left, right in zip(unique[:-1], unique[1:], strict=True)
    )
    candidates = sorted({0.0, 0.5, 1.0, *midpoints})
    results = []
    for threshold in candidates:
        metrics = classification_metrics(0.0, labels, predictions_at_threshold(scores, threshold))
        results.append((threshold, metrics))
    threshold, metrics = max(
        results,
        key=lambda item: (
            item[1].balanced_accuracy,
            item[1].f1,
            -abs(item[0] - 0.5),
            item[0],
        ),
    )
    return {
        "threshold": threshold,
        "split": "val",
        "objective": "balanced_accuracy",
        "tie_breakers": ["f1", "closest_to_0.5", "higher_threshold"],
        "candidate_count": len(candidates),
        "sample_count": len(labels),
        "metrics": {key: value for key, value in asdict(metrics).items() if key != "loss"},
    }


def threshold_from_validation_report(
    report: dict[str, Any], *, checkpoint_sha256: str, manifest_sha256: str
) -> float:
    """Read a frozen threshold only from a complete, matching validation report."""
    if not isinstance(report, dict):
        raise ValueError("threshold report must be a JSON object")
    if report.get("split") != "val" or report.get("partial") is not False:
        raise ValueError("threshold report must be a complete val report")
    count = report.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count != report.get("available_sample_count"):
        raise ValueError("threshold report has incomplete sample counts")
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != count:
        raise ValueError("threshold report must retain all validation predictions")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("threshold report checkpoint SHA256 does not match")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("threshold report manifest SHA256 does not match")
    return validate_threshold(report.get("threshold"))
