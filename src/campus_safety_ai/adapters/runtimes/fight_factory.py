from __future__ import annotations

from campus_safety_ai.adapters.runtimes.paddle_fight import PaddlePpTsmFightClassifier
from campus_safety_ai.core.fight_inference import FightClassifier
from campus_safety_ai.settings import FightModelSettings


def build_fight_classifier(settings: FightModelSettings) -> FightClassifier:
    if settings.backend == "paddle":
        classifier = PaddlePpTsmFightClassifier(settings.path)
        classifier.model_version = settings.model_version
        return classifier
    if settings.backend == "torch":
        from campus_safety_ai.adapters.runtimes.tsn_fight import TorchTsnFightClassifier

        return TorchTsnFightClassifier(settings.path, settings.device, settings.model_version)
    if settings.backend == "onnx":
        from campus_safety_ai.adapters.runtimes.tsn_fight import OnnxTsnFightClassifier

        return OnnxTsnFightClassifier(settings.path, settings.model_version)
    raise ValueError(f"unsupported fight model backend: {settings.backend}")
