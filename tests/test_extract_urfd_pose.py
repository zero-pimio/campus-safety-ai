import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise unittest.SkipTest("optional numpy dependency is not installed") from error

SPEC = importlib.util.spec_from_file_location(
    "extract_urfd_pose", Path(__file__).resolve().parents[1] / "scripts/extract_urfd_pose.py",
)
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def detections(*people):
    """Each person is (xyxy, detector confidence, distinguishable pose marker)."""
    return {
        "boxes": [person[0] for person in people],
        "scores": [person[1] for person in people],
        "keypoints": [[[person[2] + joint, person[2] + joint / 2, .9] for joint in range(17)]
                      for person in people],
    }


class PoseAssociationTests(unittest.TestCase):
    def test_same_person_continues_with_iou_and_keeps_original_pose(self):
        observations = [
            detections(([20, 10, 60, 110], .8, 20)),
            detections(([22, 10, 62, 110], .7, 22)),
            detections(([24, 10, 64, 110], .6, 24)),
        ]
        result = extract.associate_detections(observations, [0, 100, 200])
        self.assertEqual([item["association_valid"] for item in result["tracking"]], [True, True, True])
        self.assertEqual([item["track_id"] for item in result["tracking"]], [1, 1, 1])
        self.assertEqual([item["status"] for item in result["tracking"]],
                         ["initial_single_person", "associated", "associated"])
        for index, observation in enumerate(observations):
            np.testing.assert_allclose(result["keypoints"][index], observation["keypoints"][0])
            np.testing.assert_array_equal(result["boxes"][index], observation["boxes"][0])

    def test_new_high_confidence_bystander_cannot_steal_existing_identity(self):
        original = ([20, 10, 60, 110], .35, 20)
        bystander = ([150, 0, 250, 200], .99, 200)
        result = extract.associate_detections([
            detections(original), detections(bystander, ([22, 10, 62, 110], .3, 22)),
        ], [0, 100])
        second = result["tracking"][1]
        self.assertTrue(second["association_valid"])
        self.assertEqual(second["selected_index"], 1)
        self.assertEqual(second["track_id"], 1)
        self.assertEqual(result["keypoints"][1][0][0], 22)

    def test_two_similar_overlaps_are_invalid_without_moving_anchor(self):
        original = ([20, 10, 60, 110], .8, 20)
        left = ([19, 10, 59, 110], .99, 19)
        right = ([21, 10, 61, 110], .5, 21)
        result = extract.associate_detections([
            detections(original), detections(left, right), detections(original),
        ], [0, 100, 200])
        self.assertEqual(result["tracking"][1]["status"], "ambiguous_invalid")
        self.assertFalse(result["tracking"][1]["association_valid"])
        np.testing.assert_array_equal(result["keypoints"][1], np.zeros((17, 3)))
        np.testing.assert_array_equal(result["boxes"][1], np.zeros(4))
        self.assertEqual(result["tracking"][2]["status"], "associated")
        self.assertEqual(result["tracking"][2]["track_id"], 1)

    def test_disconnected_anchor_inserts_invalid_frame_before_following_new_person(self):
        first = ([20, 10, 60, 110], .8, 20)
        other = ([180, 10, 220, 110], .9, 180)
        result = extract.associate_detections([
            detections(first), detections(first), detections(other), detections(other), detections(other),
        ], [0, 100, 200, 300, 400])
        self.assertEqual([item["track_id"] for item in result["tracking"]], [1, 1, 2, 2, 2])
        self.assertEqual([item["association_valid"] for item in result["tracking"]],
                         [True, True, False, True, True])
        self.assertEqual(result["tracking"][2]["status"], "disconnected_anchor_invalid")
        # Zero confidence at the identity boundary invalidates both adjacent
        # motion pairs. The next observation can follow the new person normally.
        np.testing.assert_array_equal(np.asarray(result["keypoints"])[2, :, 2], np.zeros(17))
        np.testing.assert_array_equal(result["boxes"][2], np.zeros(4))
        np.testing.assert_allclose(result["keypoints"][3], detections(other)["keypoints"][0])
        self.assertEqual(result["tracking"][3]["status"], "associated")

    def test_missing_detection_has_zero_confidence_and_no_selected_identity(self):
        person = ([20, 10, 60, 110], .8, 20)
        result = extract.associate_detections([
            detections(person), detections(), detections(person),
        ], [0, 100, 200])
        missing = result["tracking"][1]
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["detection_count"], 0)
        self.assertIsNone(missing["selected_index"])
        self.assertIsNone(missing["track_id"])
        self.assertFalse(missing["association_valid"])
        np.testing.assert_array_equal(result["keypoints"][1], np.zeros((17, 3)))
        self.assertEqual(result["tracking"][2]["track_id"], 1)

    def test_gap_over_two_seconds_invalidates_new_anchor_but_exact_limit_is_allowed(self):
        person = detections(([20, 10, 60, 110], .8, 20))
        boundary = extract.associate_detections([person, person], [0, 2000])
        self.assertTrue(boundary["tracking"][1]["association_valid"])
        result = extract.associate_detections([person, person, person], [0, 2001, 2101])
        self.assertEqual([item["association_valid"] for item in result["tracking"]], [True, False, True])
        self.assertEqual([item["track_id"] for item in result["tracking"]], [1, 2, 2])
        self.assertEqual(result["tracking"][1]["status"], "new_anchor_invalid")
        np.testing.assert_array_equal(result["keypoints"][1], np.zeros((17, 3)))

    def test_illegal_timestamps_are_rejected(self):
        person = detections(([20, 10, 60, 110], .8, 20))
        for times in ([0], [0, 0], [100, 50], [0, float("nan")], [0, float("inf")], [-1, 0]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                extract.associate_detections([person, person], times)


if __name__ == "__main__":
    unittest.main()
