import unittest


class FallModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from campus_safety_ai.training.fall_model import FallTemporalClassifier
        except ImportError:
            raise unittest.SkipTest("optional torch/torchvision training dependencies are not installed") from None
        cls.torch = torch
        cls.factory = FallTemporalClassifier
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.torch.manual_seed(21)
        self.model = self.factory(pretrained=False)

    def test_cached_features_match_full_forward_and_head_is_trainable(self):
        torch = self.torch
        frames = torch.randn(1, 16, 3, 224, 224)
        self.model.train()
        features = self.model.forward_frame_features(frames)
        cached = self.model.forward_features(features)
        full = self.model(frames)
        self.assertEqual(tuple(features.shape), (1, 16, 576))
        self.assertEqual(tuple(full.shape), (1, 2))
        torch.testing.assert_close(cached, full, rtol=0, atol=0)
        self.assertFalse(features.requires_grad)
        full.square().sum().backward()
        self.assertIsNotNone(self.model.head.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))

    def test_temporal_summary_and_normalization_are_exact(self):
        torch = self.torch
        features = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(2, 16, 576)
        summary = self.model.summarize_features(features)
        expected = torch.cat((torch.full((2, 576), 7.5), torch.full((2, 576), 15.0),
                              torch.full((2, 576), 15.0)), dim=1)
        torch.testing.assert_close(summary, expected, rtol=0, atol=0)
        mean = torch.arange(1728, dtype=torch.float32) / 1728
        scale = torch.full((1728,), 2.0)
        self.model.set_feature_normalization(mean, scale)
        logits = self.model.forward_features(features)
        torch.testing.assert_close(logits, self.model.head((expected - mean) / scale), rtol=0, atol=0)
        mean.zero_()
        scale.zero_()
        self.assertGreater(float(self.model.feature_mean[-1]), 0)
        self.assertTrue(torch.equal(self.model.feature_scale, torch.full((1728,), 2.0)))

    def test_normalization_validation_is_atomic_and_checkpointed(self):
        torch = self.torch
        mean, scale = torch.ones(1728), torch.full((1728,), 3.0)
        self.model.set_feature_normalization(mean, scale)
        for invalid_mean, invalid_scale in (
            (torch.zeros(1, 1728), scale), (mean, torch.ones(576)),
            (torch.full((1728,), float("nan")), scale),
            (torch.zeros(1728), torch.zeros(1728)),
            (torch.zeros(1728), torch.full((1728,), -1.0)),
            (torch.zeros(1728), torch.full((1728,), float("inf"))),
        ):
            with self.subTest(mean_shape=invalid_mean.shape, scale_shape=invalid_scale.shape):
                with self.assertRaises(ValueError):
                    self.model.set_feature_normalization(invalid_mean, invalid_scale)
                torch.testing.assert_close(self.model.feature_mean, mean, rtol=0, atol=0)
                torch.testing.assert_close(self.model.feature_scale, scale, rtol=0, atol=0)
        restored = self.factory(pretrained=False)
        restored.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(restored.feature_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(restored.feature_scale, scale, rtol=0, atol=0)

    def test_backbone_stays_frozen_and_eval_when_model_trains(self):
        torch = self.torch
        self.model.eval()
        self.model.backbone.requires_grad_(True)
        self.assertIs(self.model.train(), self.model)
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.head.training)
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.backbone.parameters()))
        self.assertTrue(all(not layer.training for layer in self.model.backbone.modules()
                            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)))
        before = {name: value.clone() for name, value in self.model.backbone.named_buffers()}
        self.model.forward_frame_features(torch.randn(1, 16, 3, 224, 224))
        for name, value in self.model.backbone.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_eager_input_shapes_and_configuration_are_validated(self):
        torch = self.torch
        for shape in ((1, 15, 3, 224, 224), (1, 16, 1, 224, 224), (1, 16, 3, 112, 112), (1, 48, 224, 224)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "frames must have shape"):
                self.model.forward_frame_features(torch.empty(shape))
        for shape in ((1, 15, 576), (1, 16, 575), (16, 576)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "features must have shape"):
                self.model.forward_features(torch.empty(shape))
        for kwargs in ({"frame_count": 0}, {"num_classes": 0}, {"dropout": .2}, {"dropout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.factory(pretrained=False, **kwargs)

    def test_full_model_can_be_traced_with_fixed_16_frame_input(self):
        torch = self.torch
        self.model.eval()
        example = torch.randn(1, 16, 3, 224, 224)
        with torch.inference_mode():
            expected = self.model(example)
            traced = torch.jit.trace(self.model, example, check_trace=False)
            actual = traced(example)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
