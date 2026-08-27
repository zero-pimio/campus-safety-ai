import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "fight-video dependencies not installed")
class PaddleFightAdapterTests(unittest.TestCase):
    def test_preprocess_produces_official_model_shape(self) -> None:
        import numpy as np

        from campus_safety_ai.adapters.runtimes.paddle_fight import PaddlePpTsmFightClassifier

        frames = [np.zeros((360, 480, 3), dtype=np.uint8) for _ in range(8)]
        tensor = PaddlePpTsmFightClassifier._preprocess(frames)

        self.assertEqual(tensor.shape, (1, 8, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)

    def test_softmax_exposes_fight_class_probability(self) -> None:
        import numpy as np

        from campus_safety_ai.adapters.runtimes.paddle_fight import PaddlePpTsmFightClassifier

        probabilities = PaddlePpTsmFightClassifier._softmax(
            np.array([2.0, -1.0], dtype=np.float32)
        )

        self.assertLess(float(probabilities[1]), 0.1)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
