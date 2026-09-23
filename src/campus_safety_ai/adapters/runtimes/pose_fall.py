"""Frozen motion-head loader and conservative single-person pose inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from campus_safety_ai.contracts import FramePacket
from campus_safety_ai.core.fall_pipeline import PoseFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenPoseFallHead:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        import numpy as np
        import torch
        from campus_safety_ai.training import pose_motion

        self.checkpoint_sha256 = sha256(checkpoint)
        frozen = json.loads((checkpoint.parent / "frozen.json").read_text())
        protocol_path = checkpoint.parent / "protocol.json"
        if frozen["checkpoint_sha256"] != self.checkpoint_sha256:
            raise ValueError("checkpoint differs from frozen selection")
        if frozen["protocol_sha256"] != sha256(protocol_path):
            raise ValueError("protocol differs from frozen selection")
        protocol = json.loads(protocol_path.read_text())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["model_name"] != "pose-motion-classifier"
                or payload["feature_names"] != list(pose_motion.FEATURE_NAMES)
                or payload["feature_dim"] != len(pose_motion.FEATURE_NAMES)
                or payload["class_names"] != ["no_fall_transition", "fall_transition"]
                or payload["descriptor_source_sha256"] != sha256(Path(pose_motion.__file__))
                or payload["protocol_sha256"] != frozen["protocol_sha256"]
                or payload["threshold"] != frozen["threshold"]
                or payload["pose_checkpoint_sha256"] != protocol["pose_cache_identity"]["pose_checkpoint_sha256"]):
            raise ValueError("frozen pose classifier schema or provenance mismatch")
        self.pose_checkpoint_sha256 = payload["pose_checkpoint_sha256"]
        self.extraction_identity = protocol["pose_cache_identity"]
        self.threshold = float(payload["threshold"])
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("invalid frozen classifier threshold")
        self.model_version = f"pose-motion-{self.checkpoint_sha256[:16]}"
        self._model = pose_motion.PoseMotionClassifier(payload["feature_dim"], payload["hidden_dim"])
        self._model.load_state_dict(payload["model_state"], strict=True)
        self._model.set_feature_normalization(self._model.feature_mean, self._model.feature_scale)
        if not all(torch.isfinite(value).all() for value in self._model.state_dict().values()):
            raise ValueError("nonfinite frozen model weights")
        self._model.eval().requires_grad_(False).to(device)
        self.device, self._settings = device, payload["descriptor_settings"]

    def predict(self, frames: tuple[PoseFrame, ...]):
        import numpy as np
        import torch
        from campus_safety_ai.training.pose_motion import motion_descriptor

        origin = frames[0].observed_at
        descriptor = motion_descriptor(
            np.asarray([frame.keypoints for frame in frames]), np.asarray([frame.box for frame in frames]),
            [(frame.observed_at - origin).total_seconds() * 1000 for frame in frames], **self._settings,
        )
        if not descriptor["valid"]:
            return None, descriptor["quality"]
        with torch.inference_mode():
            inputs = torch.from_numpy(descriptor["features"][None]).to(self.device)
            score = float(self._model(inputs).softmax(dim=1)[0, 1].cpu())
        if not np.isfinite(score):
            raise ValueError("nonfinite fall score")
        return score, descriptor["quality"]


class SinglePersonPose:
    """Multiple people, missing poses or discontinuities are explicitly unknown.

    This stricter deployment guard intentionally does not claim crowd coverage.
    """

    def __init__(self, checkpoint: Path, head: FrozenPoseFallHead, device: str = "cpu"):
        if sha256(checkpoint) != head.pose_checkpoint_sha256:
            raise ValueError("pose detector differs from frozen training provenance")
        from ultralytics import YOLO

        self._model = YOLO(str(checkpoint))
        self._settings = head.extraction_identity["settings"]
        self.device = device
        self._anchor = None
        self._last_time = None
        self._stream = None
        self._track_id = 0

    def predict(self, frame: FramePacket) -> PoseFrame:
        import numpy as np

        stream = (frame.camera_id, frame.source_epoch)
        if stream != self._stream:
            self._anchor = self._last_time = None
            self._stream = stream
        result = self._model.predict(frame.image, device=self.device, imgsz=self._settings["image_size"],
                                     conf=self._settings["detection_confidence"], verbose=False, save=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) != 1 or result.keypoints is None:
            self._anchor = None
            return PoseFrame(*stream, frame.sequence, frame.captured_at, (), (), None, False,
                             "multiple_people" if len(boxes) > 1 else "missing_person")
        box = boxes[0]
        points = result.keypoints.data.cpu().numpy()[0]
        if box.shape != (4,) or points.shape != (17, 3) or not np.isfinite(box).all() or not np.isfinite(points).all():
            self._anchor = None
            return PoseFrame(*stream, frame.sequence, frame.captured_at, (), (), None, False, "invalid_pose")
        new_track = self._anchor is None
        if self._anchor is not None:
            intersection = np.maximum(np.minimum(box[2:], self._anchor[2:]) -
                                      np.maximum(box[:2], self._anchor[:2]), 0).prod()
            area = np.maximum(box[2:] - box[:2], 0).prod()
            old_area = np.maximum(self._anchor[2:] - self._anchor[:2], 0).prod()
            overlap = intersection / max(float(area + old_area - intersection), 1e-8)
            gap_ms = (frame.captured_at - self._last_time).total_seconds() * 1000
            new_track = overlap < self._settings["minimum_iou"] or gap_ms > self._settings["maximum_gap_ms"]
        if new_track:
            self._track_id += 1
        self._anchor, self._last_time = box.copy(), frame.captured_at
        return PoseFrame(*stream, frame.sequence, frame.captured_at, points, box,
                         f"{frame.camera_id}:{frame.source_epoch}:pose-{self._track_id}", True, "associated")
