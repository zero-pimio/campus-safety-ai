from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from campus_safety_ai.core.fight_inference import FightPrediction


class PaddlePpTsmFightClassifier:
    """Minimal PP-Human PP-TSM adapter for one eight-frame RGB window."""

    model_version = "pphuman-pptsm-fight-2022"
    frame_len = 8
    short_size = 340
    target_size = 320

    def __init__(self, model_dir: Path) -> None:
        model_file = model_dir / "model.pdmodel"
        params_file = model_dir / "model.pdiparams"
        missing = [path for path in (model_file, params_file) if not path.is_file()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"PP-TSM model files are missing: {names}")

        try:
            from paddle.inference import Config, create_predictor
        except ImportError as error:
            raise RuntimeError(
                "PaddlePaddle is not installed; run scripts/setup_video_runtime.sh"
            ) from error

        config = Config(str(model_file), str(params_file))
        config.disable_gpu()
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        self._predictor = create_predictor(config)

    def predict(self, rgb_frames: Sequence[Any]) -> FightPrediction:
        if len(rgb_frames) != self.frame_len:
            raise ValueError(f"PP-TSM requires exactly {self.frame_len} sampled frames")

        inputs = self._preprocess(rgb_frames)
        input_name = self._predictor.get_input_names()[0]
        input_tensor = self._predictor.get_input_handle(input_name)
        input_tensor.reshape(inputs.shape)
        input_tensor.copy_from_cpu(inputs)
        self._predictor.run()

        output_name = self._predictor.get_output_names()[0]
        output = self._predictor.get_output_handle(output_name).copy_to_cpu()
        logits = output.reshape(-1)
        if logits.size != 2:
            raise RuntimeError(f"expected two PP-TSM logits, got shape {output.shape}")

        probabilities = self._softmax(logits)
        return FightPrediction(
            logits=(float(logits[0]), float(logits[1])),
            fight_score=float(probabilities[1]),
        )

    @classmethod
    def _preprocess(cls, rgb_frames: Sequence[Any]) -> Any:
        import numpy as np
        from PIL import Image

        processed = []
        for frame in rgb_frames:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("each video frame must be an RGB HWC array")
            height, width = array.shape[:2]
            if width <= height:
                resized_width = cls.short_size
                resized_height = int(cls.short_size * 4.0 / 3.0)
            else:
                resized_height = cls.short_size
                resized_width = int(cls.short_size * 4.0 / 3.0)

            image = Image.fromarray(array.astype("uint8"), mode="RGB")
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )
            left = int(round((resized_width - cls.target_size) / 2.0))
            top = int(round((resized_height - cls.target_size) / 2.0))
            image = image.crop(
                (left, top, left + cls.target_size, top + cls.target_size)
            )
            processed.append(np.asarray(image, dtype=np.float32))

        tensor = np.stack(processed).transpose(0, 3, 1, 2) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return np.expand_dims(tensor.astype(np.float32), axis=0)

    @staticmethod
    def _softmax(logits: Any) -> Any:
        import numpy as np

        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)
