import unittest

from campus_safety_ai.training.metrics import classification_metrics, confusion_matrix


class TrainingMetricsTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix_share_the_same_predictions(self) -> None:
        labels = [0, 0, 1, 1]
        predictions = [0, 1, 0, 1]

        metrics = classification_metrics(0.5, labels, predictions)

        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.balanced_accuracy, 0.5)
        self.assertEqual(
            confusion_matrix(labels, predictions),
            {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1},
        )

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            classification_metrics(0.0, [0], [])


if __name__ == "__main__":
    unittest.main()
