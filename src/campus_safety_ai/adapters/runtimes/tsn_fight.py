from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from campus_safety_ai.core.fight_inference import FightPrediction
from campus_safety_ai.training.model import FightTsn


def preprocess_tsn_frames(
    rgb_frames: Sequence[Any], frame_count: int, image_size: int
) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

    if len(rgb_frames) != frame_count:
        raise ValueError(f"TSN requires exactly {frame_count} sampled frames")
    processed = []
    for frame in rgb_frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("each video frame must be an RGB HWC array")
        height, width = array.shape[:2]
        scale = (image_size + 32) / min(height, width)
        resized = cv2.resize(
            array,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        height, width = resized.shape[:2]
        top = (height - image_size) // 2
        left = (width - image_size) // 2
        crop = resized[top : top + image_size, left : left + image_size]
        processed.append(crop.transpose(2, 0, 1))

    tensor = np.concatenate(processed, axis=0).astype(np.float32) / 255.0
    mean = np.tile(np.array([0.485, 0.456, 0.406], dtype=np.float32), frame_count)
    std = np.tile(np.array([0.229, 0.224, 0.225], dtype=np.float32), frame_count)
    return ((tensor - mean[:, None, None]) / std[:, None, None])[None, ...]


def _softmax_fight_score(logits: Any) -> float:
    import numpy as np

    flat = np.asarray(logits, dtype=np.float32).reshape(-1)
    if flat.size != 2:
        raise RuntimeError(f"expected two TSN logits, got shape {np.asarray(logits).shape}")
    shifted = flat - np.max(flat)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return float(probabilities[1])


class TorchTsnFightClassifier:
    """Adapter for this project's MobileNetV3-Small TSN checkpoints."""

    def __init__(self, checkpoint_path: Path, device: str = "auto", model_version: str | None = None) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"fight checkpoint is missing: {checkpoint_path}")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.frame_len = int(checkpoint["frame_count"])
        self.image_size = int(checkpoint["image_size"])
        self.model_version = model_version or str(checkpoint["model_name"])
        self.device = device
        self._model = FightTsn(frame_count=self.frame_len, pretrained=False)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval().to(device)

    def predict(self, rgb_frames: Sequence[Any]) -> FightPrediction:
        import torch

        inputs = preprocess_tsn_frames(rgb_frames, self.frame_len, self.image_size)
        with torch.inference_mode():
            logits = self._model(torch.from_numpy(inputs).to(self.device)).detach().cpu().numpy().reshape(-1)
        return FightPrediction(
            logits=(float(logits[0]), float(logits[1])),
            fight_score=_softmax_fight_score(logits),
        )


class OnnxTsnFightClassifier:
    """ONNX Runtime adapter using the exporter's adjacent JSON metadata."""

    def __init__(self, model_path: Path, model_version: str | None = None) -> None:
        try:
            import json
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("install the training extra: pip install -e '.[training]'") from error

        metadata_path = model_path.with_suffix(".json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"ONNX model or metadata is missing: {model_path}, {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.frame_len = int(metadata["frame_count"])
        self.image_size = int(metadata["input_shape"][-1])
        self.model_version = model_version or str(metadata["model_name"])
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, rgb_frames: Sequence[Any]) -> FightPrediction:
        inputs = preprocess_tsn_frames(rgb_frames, self.frame_len, self.image_size)
        logits = self._session.run([self._output_name], {self._input_name: inputs})[0].reshape(-1)
        return FightPrediction(
            logits=(float(logits[0]), float(logits[1])),
            fight_score=_softmax_fight_score(logits),
        )
