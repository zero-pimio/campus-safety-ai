from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float


def classification_metrics(loss: float, labels: list[int], predictions: list[int]) -> Metrics:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions, strict=True))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions, strict=True))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions, strict=True))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(loss, accuracy, (recall + specificity) / 2, precision, recall, f1)


def confusion_matrix(labels: list[int], predictions: list[int]) -> dict[str, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    return {
        "true_negative": sum(label == prediction == 0 for label, prediction in zip(labels, predictions, strict=True)),
        "false_positive": sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions, strict=True)),
        "false_negative": sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions, strict=True)),
        "true_positive": sum(label == prediction == 1 for label, prediction in zip(labels, predictions, strict=True)),
    }
