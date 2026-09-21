from __future__ import annotations

from time import perf_counter
import math

from campus_safety_ai.contracts import BBox, Detection, Detections, FramePacket


class UltralyticsPerception:
    """Optional adapter. Install the `vision` extra before constructing it."""

    def __init__(
        self, model_path: str, model_version: str, label_map: dict[int, str] | None = None,
        *, confidence: float | None = None, device: str | None = None,
    ) -> None:
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("detection confidence must be finite and between zero and one")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("install the vision extra: pip install -e '.[vision]'") from error
        self.model = YOLO(model_path)
        self.model_version = model_version
        self.label_map = label_map
        self._predict_options = {}
        if confidence is not None:
            self._predict_options["conf"] = confidence
        if device is not None:
            self._predict_options["device"] = device

    def detect(self, frame: FramePacket) -> Detections:
        started = perf_counter()
        result = self.model.predict(frame.image, verbose=False, **self._predict_options)[0]
        names = self.label_map or result.names
        items = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                items.append(Detection(str(names[int(class_id)]), float(confidence), BBox.from_list(xyxy)))
        return Detections(
            camera_id=frame.camera_id,
            source_epoch=frame.source_epoch,
            sequence=frame.sequence,
            captured_at=frame.captured_at,
            width=frame.width,
            height=frame.height,
            detections=tuple(items),
            model_version=self.model_version,
            inference_ms=(perf_counter() - started) * 1000,
        )
