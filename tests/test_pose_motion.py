import io
import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise unittest.SkipTest("optional numpy dependency is not installed") from error

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from campus_safety_ai.training.pose_motion import FEATURE_NAMES, PoseMotionClassifier, motion_descriptor


def pose_clip(count=6):
    keypoints = np.zeros((count, 17, 3), dtype=np.float64)
    keypoints[:, :, 2] = .9
    keypoints[:, 5, :2] = [80, 80]
    keypoints[:, 6, :2] = [120, 80]
    keypoints[:, 11, :2] = [85, 150]
    keypoints[:, 12, :2] = [115, 150]
    boxes = np.tile([60, 30, 140, 210], (count, 1)).astype(np.float64)
    timestamps = np.arange(count, dtype=np.float64) * 100
    return keypoints, boxes, timestamps


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class PoseDescriptorTests(unittest.TestCase):
    def test_constant_pose_is_exactly_zero_with_no_static_values_in_features(self):
        keypoints, boxes, times = pose_clip()
        for angle in (0., .5, 1.2):
            with self.subTest(angle=angle):
                rotated = keypoints.copy()
                matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rotated[:, :, :2] = (rotated[:, :, :2] - [100, 150]) @ matrix.T + [100, 150]
                result = motion_descriptor(rotated, boxes, times)
                self.assertTrue(result["valid"])
                self.assertEqual(result["features"].dtype, np.float32)
                self.assertEqual(result["feature_names"], list(FEATURE_NAMES))
                self.assertEqual(len(FEATURE_NAMES), 32)
                np.testing.assert_array_equal(result["features"], np.zeros(32, dtype=np.float32))

    def test_global_translation_and_uniform_scaling_preserve_motion_descriptor(self):
        keypoints, boxes, times = pose_clip()
        offsets = np.arange(6) * 7.0
        keypoints[:, :, 1] += offsets[:, None]
        boxes[:, [1, 3]] += offsets[:, None]
        expected = motion_descriptor(keypoints, boxes, times)
        self.assertTrue(expected["valid"])
        transformed = keypoints.copy()
        transformed[:, :, :2] = transformed[:, :, :2] * 3.7 + [1200, -500]
        shifted_boxes = boxes * 3.7 + [1200, -500, 1200, -500]
        result = motion_descriptor(transformed, shifted_boxes, times)
        self.assertTrue(result["valid"])
        np.testing.assert_allclose(result["features"], expected["features"], rtol=1e-5, atol=1e-7)

    def test_downward_motion_and_actual_time_change_velocity_but_not_path(self):
        keypoints, boxes, times = pose_clip()
        offsets = np.arange(6) * 12.0
        keypoints[:, :, 1] += offsets[:, None]
        boxes[:, [1, 3]] += offsets[:, None]
        fast = motion_descriptor(keypoints, boxes, times)
        slow = motion_descriptor(keypoints, boxes, times * 2)
        values = dict(zip(fast["feature_names"], fast["features"], strict=True))
        slower = dict(zip(slow["feature_names"], slow["features"], strict=True))
        self.assertGreater(values["hip_vertical_max_positive_rate"], 0)
        self.assertEqual(values["hip_vertical_max_negative_rate"], 0)
        self.assertAlmostEqual(values["hip_vertical_max_positive_rate"] / 2,
                               slower["hip_vertical_max_positive_rate"])
        self.assertAlmostEqual(values["hip_vertical_total_path"], slower["hip_vertical_total_path"])
        self.assertEqual(values["hip_horizontal_total_path"], 0)

    def test_rotating_torso_produces_angular_and_relative_shape_changes(self):
        keypoints, boxes, times = pose_clip()
        for index, angle in enumerate(np.linspace(0, 1.2, len(times))):
            center = np.array([100 + 70 * np.sin(angle), 150 - 70 * np.cos(angle)])
            keypoints[index, 5, :2] = center + [-20, 0]
            keypoints[index, 6, :2] = center + [20, 0]
        result = motion_descriptor(keypoints, boxes, times)
        values = dict(zip(result["feature_names"], result["features"], strict=True))
        self.assertTrue(result["valid"])
        self.assertGreater(values["torso_angle_max_positive_rate"], 0)
        self.assertGreater(values["core_shape_total_path"], 0)
        self.assertEqual(values["hip_vertical_total_path"], 0)

    def test_confidence_is_only_a_validity_gate_and_missing_frames_are_not_bridged(self):
        keypoints, boxes, times = pose_clip(8)
        baseline = motion_descriptor(keypoints, boxes, times)
        keypoints[:, :, 2] = .5
        changed_confidence = motion_descriptor(keypoints, boxes, times)
        np.testing.assert_array_equal(changed_confidence["features"], baseline["features"])
        # Two static segments at different positions separated by an invalid frame.
        keypoints[3, 5, 2] = .1
        keypoints[4:, :, 1] += 200
        boxes[4:, [1, 3]] += 200
        result = motion_descriptor(keypoints, boxes, times)
        self.assertTrue(result["valid"])
        self.assertEqual(result["quality"]["valid_frame_count"], 7)
        self.assertEqual(result["quality"]["valid_pair_count"], 5)
        self.assertEqual(result["quality"]["valid_pair_mask"], [True, True, False, False, True, True, True])
        np.testing.assert_array_equal(result["features"], np.zeros(32, dtype=np.float32))

    def test_insufficient_core_quality_is_invalid_and_never_filled_as_normal(self):
        keypoints, boxes, times = pose_clip(7)
        keypoints[1::2, 11, 2] = .1
        result = motion_descriptor(keypoints, boxes, times)
        self.assertFalse(result["valid"])
        self.assertEqual(result["quality"]["valid_frame_count"], 4)
        self.assertEqual(result["quality"]["valid_pair_count"], 0)
        self.assertIn("fewer_than_3_valid_adjacent_pairs", result["quality"]["invalid_reasons"])
        self.assertTrue(np.isnan(result["features"]).all())
        keypoints, boxes, times = pose_clip(3)
        result = motion_descriptor(keypoints, boxes, times)
        self.assertFalse(result["valid"])
        self.assertIn("fewer_than_4_valid_frames", result["quality"]["invalid_reasons"])
        boxes[:] = np.nan
        self.assertFalse(motion_descriptor(keypoints, boxes, times)["valid"])

    def test_long_time_gaps_are_never_differenced(self):
        keypoints, boxes, _ = pose_clip()
        keypoints[3:, :, 1] += 500
        boxes[3:, [1, 3]] += 500
        result = motion_descriptor(keypoints, boxes, [0, 100, 200, 5000, 5100, 5200])
        self.assertTrue(result["valid"])
        self.assertEqual(result["quality"]["maximum_pair_gap_seconds"], 2.)
        self.assertEqual(result["quality"]["long_gap_pair_count"], 1)
        self.assertEqual(result["quality"]["valid_pair_mask"], [True, True, False, True, True])
        np.testing.assert_array_equal(result["features"], np.zeros(32, dtype=np.float32))
        invalid = motion_descriptor(keypoints, boxes, np.arange(6) * 3000)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["quality"]["valid_pair_count"], 0)
        self.assertTrue(np.isnan(invalid["features"]).all())
        keypoints, boxes, _ = pose_clip(4)
        self.assertTrue(motion_descriptor(keypoints, boxes, [0, 2000, 4000, 6000])["valid"])

    def test_timestamps_and_input_shapes_fail_fast(self):
        keypoints, boxes, times = pose_clip()
        for bad_times in (times[::-1], [0, 100, 100, 300, 400, 500], [0, 100, np.nan, 300, 400, 500]):
            with self.subTest(times=bad_times), self.assertRaisesRegex(ValueError, "strictly increasing"):
                motion_descriptor(keypoints, boxes, bad_times)
        with self.assertRaisesRegex(ValueError, "shape"):
            motion_descriptor(keypoints[:, :16], boxes, times)
        with self.assertRaisesRegex(ValueError, "align"):
            motion_descriptor(keypoints, boxes[:-1], times)
        with self.assertRaisesRegex(ValueError, "min_conf"):
            motion_descriptor(keypoints, boxes, times, min_conf=np.nan)


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class PoseClassifierTests(unittest.TestCase):
    def test_linear_and_mlp_normalization_trainable_head_and_checkpoint(self):
        torch.manual_seed(5)
        for hidden_dim in (0, 16):
            with self.subTest(hidden_dim=hidden_dim):
                model = PoseMotionClassifier(32, hidden_dim)
                mean, scale = torch.arange(32) / 32, torch.full((32,), 2.)
                model.set_feature_normalization(mean, scale)
                features = torch.randn(3, 32)
                logits = model(features)
                self.assertEqual(tuple(logits.shape), (3, 2))
                torch.testing.assert_close(logits, model.head((features - mean) / scale), rtol=0, atol=0)
                logits.square().mean().backward()
                self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
                self.assertFalse(model.feature_mean.requires_grad)
                clone = PoseMotionClassifier(32, hidden_dim)
                clone.load_state_dict(model.state_dict())
                torch.testing.assert_close(clone(features), logits)

    def test_classifier_rejects_bad_shapes_and_normalization_atomically(self):
        model = PoseMotionClassifier(32)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.ones(32))
        for mean, scale in ((torch.ones(31), torch.ones(32)), (torch.ones(32), torch.zeros(32)),
                            (torch.ones(32), torch.full((32,), float("inf")))):
            with self.subTest(mean_shape=mean.shape), self.assertRaises(ValueError):
                model.set_feature_normalization(mean, scale)
            torch.testing.assert_close(model.feature_mean, torch.zeros(32), rtol=0, atol=0)
            torch.testing.assert_close(model.feature_scale, torch.ones(32), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            PoseMotionClassifier(32, 8)

    def test_onnx_export_matches_torch_for_both_supported_heads(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("optional ONNX dependencies are not installed")
        for hidden_dim in (0, 16):
            with self.subTest(hidden_dim=hidden_dim):
                model = PoseMotionClassifier(32, hidden_dim).eval()
                model.set_feature_normalization(torch.linspace(-1, 1, 32), torch.linspace(.1, 2, 32))
                example = torch.randn(1, 32)
                output = io.BytesIO()
                torch.onnx.export(model, example, output, input_names=["motion_features"], output_names=["logits"],
                                  opset_version=17, dynamo=False)
                graph = onnx.load_model_from_string(output.getvalue())
                onnx.checker.check_model(graph)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(output.getvalue(), sess_options=options, providers=["CPUExecutionProvider"])
                actual = session.run(["logits"], {"motion_features": example.numpy()})[0]
                with torch.inference_mode():
                    expected = model(example).numpy()
                self.assertEqual(actual.shape, (1, 2))
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
